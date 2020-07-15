// code adapted from https://github.com/JakobEngel/dso
//  g++ main2.cpp -fPIC -shared -o libtest.so -std=c++0x

#include <iostream>
#include <string>
#include <stdlib.h>
#include <cmath>
#include <cstring>
#include <assert.h>
using namespace std;

// global calib
#define PYR_LEVELS 5

int wG[PYR_LEVELS], hG[PYR_LEVELS];
int pyrLevelsUsed;

// pixel selector
enum PixelSelectorStatus {PIXSEL_VOID=0, PIXSEL_1, PIXSEL_2, PIXSEL_3};
int currentPotential;
bool allowFast;
unsigned char* randomPattern;
int* gradHist;
float* ths;
float* thsSmoothed;
int thsStep;
float* gradHistFrame;

float setting_minGradHistCut = 0.5;
int setting_minGradHistAdd = 7;
float setting_gradDownweightPerLevel = 0.75;
bool setting_selectDirectionDistribution = false;

// framehessian for make images
float* dI;
float* dIp[PYR_LEVELS];
float* absSquaredGrad[PYR_LEVELS];

//inline ~FrameHessian()
//{
//    for(int i=0;i<pyrLevelsUsed;i++)
//    {
//        delete[] dIp[i];
//        delete[]  absSquaredGrad[i];
//
//    }
//};

void setGlobalCalib(int w, int h)
{
    int wlvl=w;
    int hlvl=h;
    pyrLevelsUsed=1;
    while(wlvl%2==0 && hlvl%2==0 && wlvl*hlvl > 5000 && pyrLevelsUsed < PYR_LEVELS)
    {
        wlvl /=2;
        hlvl /=2;
        pyrLevelsUsed++;
    }
//    printf("using pyramid levels 0 to %d. coarsest resolution: %d x %d!\n",
//           pyrLevelsUsed-1, wlvl, hlvl);
//    if(wlvl>100 && hlvl > 100)
//    {
//        printf("\n\n===============WARNING!===================\n "
//               "using not enough pyramid levels.\n"
//               "Consider scaling to a resolution that is a multiple of a power of 2.\n");
//    }
//    if(pyrLevelsUsed < 3)
//    {
//        printf("\n\n===============WARNING!===================\n "
//               "I need higher resolution.\n"
//               "I will probably segfault.\n");
//    }

    wG[0] = w;
    hG[0] = h;

    for (int level = 1; level < pyrLevelsUsed; ++ level)
    {
        wG[level] = w >> level;
        hG[level] = h >> level;
    }
}

// todo don't call this every time
void initPixelSelector(int w, int h)
{
    randomPattern = new unsigned char[w*h];
    std::srand(3141592);	// want to be deterministic.
    for(int i=0;i<w*h;i++) randomPattern[i] = rand() & 0xFF;

    currentPotential=3;


    gradHist = new int[100*(1+w/32)*(1+h/32)];
    ths = new float[(w/32)*(h/32)+100];
    thsSmoothed = new float[(w/32)*(h/32)+100];

    allowFast=false;
    gradHistFrame = 0;
}

void deleteSelector()
{
    delete[] randomPattern;
    delete[] gradHist;
    delete[] ths;
    delete[] thsSmoothed;
}

void deleteImages()
{
    for(int i=0;i<pyrLevelsUsed;i++)
    {
        delete[] dIp[i];
        delete[] absSquaredGrad[i];
    }
}

void makeImages(float* color)
{
    for(int i=0;i<pyrLevelsUsed;i++)
    {
        // may need free before allocate
        // std::cout << "make images" << std::endl;
        dIp[i] = new float[wG[i]*hG[i]*3];
        absSquaredGrad[i] = new float[wG[i]*hG[i]];
    }
    dI = dIp[0];


    // make d0
    int w=wG[0];
    int h=hG[0];
    for(int i=0;i<w*h;i++) {
        dI[i * 3] = color[i];
        // std::cout << dI[i*3] << endl;
    }

    for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
    {
        int wl = wG[lvl], hl = hG[lvl];
        //todo
        float* dI_l = dIp[lvl];

        float* dabs_l = absSquaredGrad[lvl];
        if(lvl>0)
        {
            int lvlm1 = lvl-1;
            int wlm1 = wG[lvlm1];
            float* dI_lm = dIp[lvlm1];

            for(int y=0;y<hl;y++)
                for(int x=0;x<wl;x++)
                {
                    dI_l[(x + y*wl)*3] = 0.25f * (dI_lm[(2*x   + 2*y*wlm1)*3] +
                                                 dI_lm[(2*x+1 + 2*y*wlm1)*3] +
                                                 dI_lm[(2*x   + 2*y*wlm1+wlm1)*3] +
                                                 dI_lm[(2*x+1 + 2*y*wlm1+wlm1)*3]);
                }
        }

        for(int idx=wl;idx < wl*(hl-1);idx++)
        {
            float dx = 0.5f*(dI_l[(idx+1)*3] - dI_l[(idx-1)*3]);
            float dy = 0.5f*(dI_l[(idx+wl)*3] - dI_l[(idx-wl)*3]);


            if(!std::isfinite(dx)) dx=0;
            if(!std::isfinite(dy)) dy=0;

            dI_l[idx*3+1] = dx;
            dI_l[idx*3+2] = dy;


            dabs_l[idx] = dx*dx+dy*dy;
        }
    }
}


int computeHistQuantil(int* hist, float below)
{
    int th = hist[0]*below+0.5f;
    for(int i=0;i<90;i++)
    {
        th -= hist[i+1];
        if(th<0) return i;
    }
    return 90;
}


void makeHists(float* image)
{
    gradHistFrame = image;

    float * mapmax0 = absSquaredGrad[0];

    int w = wG[0];
    int h = hG[0];

    int w32 = w/32;
    int h32 = h/32;
    thsStep = w32;

    for(int y=0;y<h32;y++)
        for(int x=0;x<w32;x++)
        {
            float* map0 = mapmax0+32*x+32*y*w;
            int* hist0 = gradHist;// + 50*(x+y*w32);
            memset(hist0,0,sizeof(int)*50);

            for(int j=0;j<32;j++) for(int i=0;i<32;i++)
                {
                    int it = i+32*x;
                    int jt = j+32*y;
                    if(it>w-2 || jt>h-2 || it<1 || jt<1) continue;
                    int g = sqrtf(map0[i+j*w]);
                    if(g>48) g=48;
                    hist0[g+1]++;
                    hist0[0]++;
                }

            ths[x+y*w32] = computeHistQuantil(hist0,setting_minGradHistCut) + setting_minGradHistAdd;
        }

    for(int y=0;y<h32;y++)
        for(int x=0;x<w32;x++)
        {
            float sum=0,num=0;
            if(x>0)
            {
                if(y>0) 	{num++; 	sum+=ths[x-1+(y-1)*w32];}
                if(y<h32-1) {num++; 	sum+=ths[x-1+(y+1)*w32];}
                num++; sum+=ths[x-1+(y)*w32];
            }

            if(x<w32-1)
            {
                if(y>0) 	{num++; 	sum+=ths[x+1+(y-1)*w32];}
                if(y<h32-1) {num++; 	sum+=ths[x+1+(y+1)*w32];}
                num++; sum+=ths[x+1+(y)*w32];
            }

            if(y>0) 	{num++; 	sum+=ths[x+(y-1)*w32];}
            if(y<h32-1) {num++; 	sum+=ths[x+(y+1)*w32];}
            num++; sum+=ths[x+y*w32];

            thsSmoothed[x+y*w32] = (sum/num) * (sum/num);

        }

}

void select_point(float* fh,
                  float* map_out, int pot, float thFactor, int* out_n);

int makeMaps(
        float *fh,
        float* map_out, float density, int recursionsLeft, bool plot, float thFactor)
{
    float numHave=0;
    float numWant=density;
    float quotia;
    int idealPotential = currentPotential;


    {
        // the number of selected pixels behaves approximately as
        // K / (pot+1)^2, where K is a scene-dependent constant.
        // we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.

        if(fh != gradHistFrame) makeHists(fh);

        // select!
        int n[3];
        select_point(fh, map_out,currentPotential, thFactor, n);
        // cout << "after select points" << n[0] << " " << n[1] << " " << n[2] << " " << endl;

        // sub-select!
        numHave = n[0]+n[1]+n[2];
        quotia = numWant / numHave;

        // by default we want to over-sample by 40% just to be sure.
        float K = numHave * (currentPotential+1) * (currentPotential+1);
        idealPotential = sqrtf(K/numWant)-1;	// round down.
        if(idealPotential<1) idealPotential=1;

        if( recursionsLeft>0 && quotia > 1.25 && currentPotential>1)
        {
            //re-sample to get more points!
            // potential needs to be smaller
            if(idealPotential>=currentPotential)
                idealPotential = currentPotential-1;

            //		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
            //				100*numHave/(float)(wG[0]*hG[0]),
            //				100*numWant/(float)(wG[0]*hG[0]),
            //				currentPotential,
            //				idealPotential);
            currentPotential = idealPotential;
            return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);
        }
        else if(recursionsLeft>0 && quotia < 0.25)
        {
            // re-sample to get less points!

            if(idealPotential<=currentPotential)
                idealPotential = currentPotential+1;

            //		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
            //				100*numHave/(float)(wG[0]*hG[0]),
            //				100*numWant/(float)(wG[0]*hG[0]),
            //				currentPotential,
            //				idealPotential);
            currentPotential = idealPotential;
            return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);

        }
    }

    int numHaveSub = numHave;
    if(quotia < 0.95)
    {
        int wh=wG[0]*hG[0];
        int rn=0;
        unsigned char charTH = 255*quotia;
        for(int i=0;i<wh;i++)
        {
            if(map_out[i] != 0)
            {
                if(randomPattern[rn] > charTH )
                {
                    map_out[i]=0;
                    numHaveSub--;
                }
                rn++;
            }
        }
    }

//	printf("PixelSelector: have %.2f%%, need %.2f%%. KEEPCURR with pot %d -> %d. Subsampled to %.2f%%\n",
//			100*numHave/(float)(wG[0]*hG[0]),
//			100*numWant/(float)(wG[0]*hG[0]),
//			currentPotential,
//			idealPotential,
//			100*numHaveSub/(float)(wG[0]*hG[0]));
    currentPotential = idealPotential;

    return numHaveSub;
}


inline float norm(float* v)
{
    return std::sqrt(v[1] * v[1] + v[2] * v[2]);
}

void select_point(float* fh,
            float* map_out, int pot, float thFactor, int* out_n)
{

    float* map0 = dI;

    float * mapmax0 = absSquaredGrad[0];
    float * mapmax1 = absSquaredGrad[1];
    float * mapmax2 = absSquaredGrad[2];


    int w = wG[0];
    int w1 = wG[1];
    int w2 = wG[2];
    int h = hG[0];

    memset(map_out,0,w*h*sizeof(PixelSelectorStatus));

    float dw1 = setting_gradDownweightPerLevel;
    float dw2 = dw1*dw1;


    int n3=0, n2=0, n4=0;
    for(int y4=0;y4<h;y4+=(4*pot)) for(int x4=0;x4<w;x4+=(4*pot))
        {
            int my3 = std::min((4*pot), h-y4);
            int mx3 = std::min((4*pot), w-x4);
            int bestIdx4=-1; float bestVal4=0;
            // Vec2f dir4 = directions[randomPattern[n2] & 0xF];
            for(int y3=0;y3<my3;y3+=(2*pot)) for(int x3=0;x3<mx3;x3+=(2*pot))
                {
                    int x34 = x3+x4;
                    int y34 = y3+y4;
                    int my2 = std::min((2*pot), h-y34);
                    int mx2 = std::min((2*pot), w-x34);
                    int bestIdx3=-1; float bestVal3=0;
                    // Vec2f dir3 = directions[randomPattern[n2] & 0xF];
                    for(int y2=0;y2<my2;y2+=pot) for(int x2=0;x2<mx2;x2+=pot)
                        {
                            int x234 = x2+x34;
                            int y234 = y2+y34;
                            int my1 = std::min(pot, h-y234);
                            int mx1 = std::min(pot, w-x234);
                            int bestIdx2=-1; float bestVal2=0;
                            // Vec2f dir2 = directions[randomPattern[n2] & 0xF];
                            for(int y1=0;y1<my1;y1+=1) for(int x1=0;x1<mx1;x1+=1)
                                {
                                    assert(x1+x234 < w);
                                    assert(y1+y234 < h);
                                    int idx = x1+x234 + w*(y1+y234);
                                    int xf = x1+x234;
                                    int yf = y1+y234;

                                    if(xf<4 || xf>=w-5 || yf<4 || yf>h-4) continue;


                                    float pixelTH0 = thsSmoothed[(xf>>5) + (yf>>5) * thsStep];
                                    float pixelTH1 = pixelTH0*dw1;
                                    float pixelTH2 = pixelTH1*dw2;


                                    float ag0 = mapmax0[idx];
                                    if(ag0 > pixelTH0*thFactor)
                                    {
                                        // Vec2f ag0d = map0[idx].tail<2>();
                                        // float dirNorm = fabsf((float)(ag0d.dot(dir2)));
                                        // if(!setting_selectDirectionDistribution) dirNorm = ag0;

                                        // float dirNorm = norm(map0[idx*3]);
                                        float dirNorm = ag0;
                                        if(dirNorm > bestVal2)
                                        { bestVal2 = dirNorm; bestIdx2 = idx; bestIdx3 = -2; bestIdx4 = -2;}
                                    }
                                    if(bestIdx3==-2) continue;

                                    float ag1 = mapmax1[(int)(xf*0.5f+0.25f) + (int)(yf*0.5f+0.25f)*w1];
                                    if(ag1 > pixelTH1*thFactor)
                                    {
                                        // Vec2f ag0d = map0[idx].tail<2>();
                                        // float dirNorm = fabsf((float)(ag0d.dot(dir3)));
                                        // if(!setting_selectDirectionDistribution) dirNorm = ag1;

                                        // float dirNorm = norm(map0[idx*3]);
                                        float dirNorm = ag1;
                                        if(dirNorm > bestVal3)
                                        { bestVal3 = dirNorm; bestIdx3 = idx; bestIdx4 = -2;}
                                    }
                                    if(bestIdx4==-2) continue;

                                    float ag2 = mapmax2[(int)(xf*0.25f+0.125) + (int)(yf*0.25f+0.125)*w2];
                                    if(ag2 > pixelTH2*thFactor)
                                    {
                                        // Vec2f ag0d = map0[idx].tail<2>();
                                        // float dirNorm = fabsf((float)(ag0d.dot(dir4)));
                                        // if(!setting_selectDirectionDistribution) dirNorm = ag2;

                                        // float dirNorm = norm(map0[idx*3]);
                                        float dirNorm = ag2;
                                        if(dirNorm > bestVal4)
                                        { bestVal4 = dirNorm; bestIdx4 = idx; }
                                    }
                                }

                            if(bestIdx2>0)
                            {
                                map_out[bestIdx2] = 1;
                                bestVal3 = 1e10;
                                n2++;
                            }
                        }

                    if(bestIdx3>0)
                    {
                        map_out[bestIdx3] = 2;
                        bestVal4 = 1e10;
                        n3++;
                    }
                }

            if(bestIdx4>0)
            {
                map_out[bestIdx4] = 4;
                n4++;
            }
        }

    out_n[0] = n2;
    out_n[1] = n3;
    out_n[2] = n4;
//    return Eigen::Vector3i(n2,n3,n4);
    return;
}



int main(int w, int h, float* image, int num, int max_num, float* result) {

    // std::cout << "w: " << w << " h: " << h << " num: " << num << std::endl;

    setGlobalCalib(w, h);
    initPixelSelector(w, h);
    // todo free memory
    makeImages(image);

    float* map_out = new float[h*w];

    int numPointsTotal = makeMaps(image, map_out, 2000, 3, false, 1);

    // std::cout << "total points " << numPointsTotal << std::endl;
    int n = 0;
    for (int i = 0; i < h; ++i) for (int j = 0; j < w; ++j)
    {
        if (map_out[i*w + j] != 0){
//            std::cout << i << " " << j << endl;
            result[n*2] = i;
            result[n*2+1] = j;
            n += 1;
            if (n > max_num) break;
        }
    }
    // todo free memory
    delete []map_out;
    deleteImages();
    deleteSelector();
    return numPointsTotal;
}
