#include <algorithm>
#include<cmath>
#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<iostream>
#include<chrono>
#include<fstream>
using namespace std;


#define EXTLOOP 1
#define LOOP 10000
#define TIME 1e6

#define REAL double
using namespace std;
typedef std::chrono::high_resolution_clock Time;

/*
 * Generates some random floating point numbers in double precision inside chosen range.
 * Try a bunch of exp calls on these numbers, dumping speedup from different precision/intrinsics used
 * Dumping also the relative errors compare to the result in double precision.
 * Try scalar and vectorize version.
 * GCC, INTEL intrinsics.
 * */

static const string OFILE = "None";
static const unsigned int SEED = 123456789;
void foolCompiler(double * arr);

struct stats{
    double m;
    double M;
    double stddev;
    double mean;
    double med;
};
#define EPSILON 1E-25
double relError(double r, double b){
    double relErr = 0.;
    if(r < EPSILON && b > EPSILON)
        return 1e31;//Arbitrary infinite
    if(r < EPSILON && b < EPSILON)
        return NAN;
    if(r > EPSILON && b < EPSILON)
        return 0.;
    relErr = abs(b-r) / abs(r);
    return relErr;
}
double mean_(double * a, unsigned int n){
    // Compute mean (average of elements)
    double sum = 0;
    for (unsigned int i = 0; i < n; i++)
        sum += a[i];
    double mean = (double)sum /
        (double)n;
    return mean;
}
double variance(double a[], unsigned int n)
{

    // Compute sum squared
    // differences with mean.
    double mean = mean_(a,n);
    double sqDiff = 0;
    for (unsigned int i = 0; i < n; i++)
        sqDiff += (a[i] - mean) *
            (a[i] - mean);
    return sqDiff / n;
}

double standardDeviation(double arr[],
        int n)
{
    return sqrt(variance(arr, n));
}

template <typename T>
struct stats relativeErrorStats(double * refer, T * _comp, unsigned int size){
    struct stats relerrStats;
    double * comp;
    if constexpr (std::is_same<T,float>::value){
        comp  = new double(size);
        for(int i=0; i<size ;i++)
            comp[i] = (double) _comp[i];
    }else if constexpr (std::is_same<T,double>::value){
        comp = _comp;
    }

    relerrStats.m = *min_element(comp, comp+size);
    relerrStats.M = *max_element(comp, comp+size);
    relerrStats.mean = mean_(comp,size);
    relerrStats.stddev = standardDeviation(comp, size);
    sort(comp, comp+size);
    relerrStats.med = comp[(unsigned int) (size/2)];
    if constexpr (std::is_same<T,float>::value){
        delete []comp;
    }
    return relerrStats;
}

REAL * generateArray(string outputfile, unsigned int prec, unsigned long size, long long rangeinf, long long rangesup){
    srand(SEED);
#ifdef USE_INTEL_COMPILER
    REAL * arr = (REAL*) _mm_malloc(size*sizeof(REAL), 64);
#else
    REAL * arr = (REAL*)aligned_alloc(64,size*sizeof(REAL));
#endif
    cout.precision(prec);

    std::ofstream ofile;
    ofile.open(outputfile, ios::app); //app is append which means it will put the text at the end
    double e;
    for(int i=0;i<size;i++){
        e = (double)rand()/(double)RAND_MAX; // (x/N in [0;1])
        e = rangeinf + e*(rangesup - rangeinf); // x--> (b-a)*x +a [0:1] --> [a;b]
        /* Random number generated between 2**R and 2**(R+1) */
        arr[i] = e;
        ofile << e << endl;
    }
    return arr;
}

int main(int argc, char * argv[]){
    if(argc < 5){
        cerr << "Not enough args\n SIZE (size of the array containing exp inputs)\
            \n RANGEinf RANGEsup (random generation between 2**Rinf and 2**Rsup)\
            \n Input file: either to drop result (if it does not exist yet) or to read and compare results against,\
            \n Optional: input file containing random input naumbers." << endl;
        exit(-1);
    }
    double * inputVal;
    double * doublePrec;
    float  * singlePrecOutput;
    double * singlePrec2Cast;
    double * singlePrec1CastOutput;
    float  * singlePrec1CastInput;
    float  * inputValSinglePrec;
    float  * singlePrec;

    unsigned int SIZE = (unsigned int)atoi(argv[1]);
    long long RANGEinf = (long long)atoll(argv[2]);
    long long RANGEsup = (long long)atoll(argv[3]);

    string inputfile("None");
    string resultfile("RefResults.dat");
    doublePrec = new double[SIZE];
    singlePrec2Cast = new double[SIZE];
    singlePrec1CastOutput= new double[SIZE];
#ifdef USE_INTEL_COMPILER
    singlePrec            = (float*) _mm_malloc(SIZE*sizeof(float), 64);
    singlePrecOutput      = (float*) _mm_malloc(SIZE*sizeof(float), 64);
    singlePrec1CastInput  = (float*) _mm_malloc(SIZE*sizeof(float), 64);
    inputValSinglePrec    = (float*) _mm_malloc(SIZE*sizeof(float), 64);
    doublePrec            = (double*) _mm_malloc(SIZE*sizeof(double), 64);
    singlePrec2Cast       = (double*) _mm_malloc(SIZE*sizeof(double), 64);
    singlePrec1CastOutput = (double*) _mm_malloc(SIZE*sizeof(double), 64);
#else
    singlePrec            = (float*)aligned_alloc(64,SIZE*sizeof(float));
    singlePrecOutput      = (float*)aligned_alloc(64,SIZE*sizeof(float));
    singlePrec1CastInput  = (float*)aligned_alloc(64,SIZE*sizeof(float));
    inputValSinglePrec    = (float*)aligned_alloc(64,SIZE*sizeof(float));
    doublePrec            = (double*)aligned_alloc(64,SIZE*sizeof(double));
    singlePrec2Cast       = (double*)aligned_alloc(64,SIZE*sizeof(double));
    singlePrec1CastOutput = (double*)aligned_alloc(64,SIZE*sizeof(double));
#endif

    resultfile = string(argv[4]);
    // dump result in Reference file or read from reference file to compare results to.

    // Generate new random numbers OR read existing file
    if (argc == 6)
        inputfile = string(argv[5]);
    if (inputfile.compare("None") != 0){
        std::ifstream ifile;
        ifile.open(inputfile, std::ios::in);
#ifdef USE_INTEL_COMPILER
        inputVal = (double*) _mm_malloc(SIZE*sizeof(double), 64);
#else
        inputVal = (double*) aligned_alloc(64,SIZE*sizeof(double));
#endif
        double e;
        for (int i=0;i<SIZE;i++){
            ifile >> e;
            inputVal[i] = e;
        }
    }else{//inputfile == None
        cerr << "Generating ..."<<endl;
        inputVal = generateArray(string("numbers-") + to_string(RANGEinf) + string("-") + to_string(RANGEsup) + string(".dat"), 15, SIZE, RANGEinf, RANGEsup);
    }
    for(int i = 0; i < SIZE; i++)
        inputValSinglePrec[i] = (float) inputVal[i];

    for(int k = 0; k< EXTLOOP; k++){
        auto doublePrecTime = 0;
        auto singlePrecNoCastTime = 0;
        auto singlePrec2CastTime = 0;
        auto singlePrec1CastInputTime = 0;
        auto singlePrec1CastOutputTime = 0;
        for(int j = 0; j< LOOP; j++){

            // Warm up
            for (int i=0;i<SIZE;i++){
                singlePrecOutput[i] = expf(inputValSinglePrec[i]);
            }
            // single precision NO cast
            auto t0 = Time::now();
            for (int i=0;i<SIZE;i++){
                singlePrecOutput[i] = expf(inputValSinglePrec[i]);
            }
            auto t1 = Time::now();
            singlePrecNoCastTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

            // Warm up
            for (int i=0;i<SIZE;i++){
                doublePrec[i] = exp(inputVal[i]);
            }
            // double precision
            t0 = Time::now();
            for (int i=0;i<SIZE;i++){
                doublePrec[i] = exp(inputVal[i]);
            }
            t1 = Time::now();
            doublePrecTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

            // Warm up
            for (int i=0;i<SIZE;i++){
                singlePrec2Cast[i] = (double) expf((float)inputVal[i]);
            }
            // single precision 2 casts
            t0 = Time::now();
            for (int i=0;i<SIZE;i++){
                singlePrec2Cast[i] = (double) expf((float)inputVal[i]);
            }
            t1 = Time::now();
            singlePrec2CastTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

            // Warmup
            for (int i=0;i<SIZE;i++){
                singlePrec1CastInput[i] = expf((float)inputVal[i]);
            }
            // single precision 1 cast in
            t0 = Time::now();
            for (int i=0;i<SIZE;i++){
                singlePrec1CastInput[i] = expf((float)inputVal[i]);
            }
            t1 = Time::now();
            singlePrec1CastInputTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

            //Warm up
            for (int i=0;i<SIZE;i++){
                singlePrec1CastOutput[i] = (double) expf(inputValSinglePrec[i]);
            }
            // single precision 1 cast out
            t0 = Time::now();
            for (int i=0;i<SIZE;i++){
                singlePrec1CastOutput[i] = (double) expf(inputValSinglePrec[i]);
            }
            t1 = Time::now();
            singlePrec1CastOutputTime+= std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        }
        // Display of time + relative error compute
        struct stats relErrStats2Cast = relativeErrorStats<double>(doublePrec, singlePrec2Cast, SIZE);
        struct stats relErrStats1CastIn = relativeErrorStats<float>(doublePrec, singlePrec1CastInput, SIZE);
        struct stats relErrStats1CastOut = relativeErrorStats<double>(doublePrec, singlePrec1CastOutput, SIZE);
        struct stats relErrStatsNoCast = relativeErrorStats<float>(doublePrec, singlePrecOutput, SIZE);
        cerr.precision(2);
        cerr << "time(ns) RelErrMean med stddev min MAX" <<endl;
        cerr << "doublePrec singlePrec2casts singlePrec1CastIn sinblePrec1castOUt singlePrecNoCast" << endl;
        cerr << (double)doublePrecTime                                                                          / TIME     << "\t"
             << (double)singlePrec2CastTime                                                                     / TIME     << "\t"
             << (double)singlePrec1CastInputTime                                                                / TIME     << "\t"
             << (double)singlePrec1CastOutputTime                                                               / TIME     << "\t"
             << (double)singlePrecNoCastTime                                                                    / TIME     << endl;

        cerr << relErrStats2Cast.mean                                                                          << "\t"
             << relErrStats1CastIn.mean                                                                     << "\t"
             << relErrStats1CastOut.mean                                                                    << "\t"
             << relErrStatsNoCast.mean                                                                         << endl;

        cerr << relErrStats2Cast.med                                                                          << "\t"
             << relErrStats1CastIn.med                                                                     << "\t"
             << relErrStats1CastOut.med                                                                    << "\t"
             << relErrStatsNoCast.med                                                                         << endl;

        cerr << relErrStats2Cast.stddev                                                                          << "\t"
             << relErrStats1CastIn.stddev                                                                     << "\t"
             << relErrStats1CastOut.stddev                                                                    << "\t"
             << relErrStatsNoCast.stddev                                                                         << endl;

        cerr << relErrStats2Cast.m                                                                          << "\t"
             << relErrStats1CastIn.m                                                                     << "\t"
             << relErrStats1CastOut.m                                                                    << "\t"
             << relErrStatsNoCast.m                                                                         << endl;

        cerr << relErrStats2Cast.M                                                                          << "\t"
             << relErrStats1CastIn.M                                                                     << "\t"
             << relErrStats1CastOut.M                                                                    << "\t"
             << relErrStatsNoCast.M                                                                         << endl;

        foolCompiler(doublePrec);
        //std::cerr << "Expf("<<singlePrec[SIZE-1]<<") float: " << singlePrecTime/1000. << " ns " << inputVal[SIZE-1]<< " with cast cost." << std::endl;
        //std::cerr << "Expf("<<singlePrec[SIZE-1]<<") float: " << singlePrecTime/1000. << " ns " << inputValSinglePrec[SIZE-1]<< " without cast cost." << std::endl;
        //std::cerr << "Exp("<< doublePrec[SIZE-1]<<") double: " << doublePrecTime/1000. << " ns " << inputVal[SIZE-1]<<std::endl
    }
    return 0;
}
