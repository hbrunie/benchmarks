#include <algorithm>
#include<cmath>
#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<iostream>
#include<chrono>
#include<fstream>
using namespace std;


#define EXTLOOP 10
#define LOOP 10000

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
struct stats relativeErrorStats(double * refer, T * comp, unsigned int size){
    struct stats relerrStats;

    relerrStats.m = *min_element(comp, comp+size);
    relerrStats.M = *max_element(comp, comp+size);
    relerrStats.mean = mean_(comp,size);
    relerrStats.stddev = standardDeviation(comp, size);
    sort(comp, comp+size);
    relerrStats.med = comp[(unsigned int) (size/2)];

    return relerrStats;
}

REAL * generateArray(string outputfile, unsigned int prec, unsigned long size, long long rangeinf, long long rangesup){
    srand(SEED);
    REAL * arr = new REAL[size];
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
    if(argc != 4){
        cerr << "Not enough args\n SIZE (size of the array containing exp inputs)\
            \n RANGEinf RANGEsup (random generation between 2**Rinf and 2**Rsup)\
            \n Optional: input file containing randon input numbers." << endl;
        exit(-1);
    }
    double * inputVal;
    double * doublePrec;
    double * singlePrec2Cast;
    double * singlePrec1CastOutput;
    float * inputValSinglePrec;
    float * singlePrec;

    unsigned int SIZE = (unsigned int)atoi(argv[1]);
    long long RANGEinf = (long long)atoll(argv[2]);
    long long RANGEsup = (long long)atoll(argv[3]);

    string inputfile("None");
    doublePrec = new double[SIZE];
    singlePrec = new float[SIZE];
    singlePrec2Cast = new double[SIZE];
    singlePrec1CastOutput= new double[SIZE];
    inputValSinglePrec = new float[SIZE];

    if (argc == 5)
        inputfile = string(argv[4]);

    if (inputfile.compare("None") != 0){
        std::ifstream ifile;
        ifile.open(inputfile, std::ios::in);
        inputVal = new double[SIZE];
        double e;
        for (int i=0;i<SIZE;i++){
            ifile >> e;
            inputVal[i] = e;
        }
    }else{//inputfile == None
        cerr << "Generating ..."<<endl;
        inputVal = generateArray(string("numbers.dat"), 15, SIZE, RANGEinf, RANGEsup);
    }
    for(int i = 0; i < SIZE; i++)
        inputValSinglePrec[i] = (float) inputVal[i];

    for(int k = 0; k< EXTLOOP; k++){
        auto doublePrecTime = 0;
        auto singlePrec2CastTime = 0;
        auto singlePrec1CastInputTime = 0;
        auto singlePrec1CastOutputTime = 0;
        for(int j = 0; j< LOOP; j++){
            for (int i=0;i<SIZE;i++){
                doublePrec[i] = exp(inputVal[i]);
            }
            auto t0 = Time::now();
            for (int i=0;i<SIZE;i++){
                doublePrec[i] = exp(inputVal[i]);
            }
            auto t1 = Time::now();
            doublePrecTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

            for (int i=0;i<SIZE;i++){
                singlePrec2Cast[i] = (double) expf((float)inputVal[i]);
            }
            t0 = Time::now();
            for (int i=0;i<SIZE;i++){
                singlePrec2Cast[i] = (double) expf((float)inputVal[i]);
            }
            t1 = Time::now();
            singlePrec2CastTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

            //t0 = Time::now();
            //for (int i=0;i<SIZE;i++){
            //    singlePrec1CastInput[i] = expf((float)inputVal[i]);
            //}
            //t1 = Time::now();
            //singlePrec1CastInputTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

            t0 = Time::now();
            for (int i=0;i<SIZE;i++){
                singlePrec1CastOutput[i] = (double) expf(inputValSinglePrec[i]);
            }
            t1 = Time::now();
            singlePrec1CastOutputTime+= std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        }
        struct stats relErrStats = relativeErrorStats<double>(doublePrec, singlePrec2Cast, SIZE);
        cerr.precision(15);
        cerr << "single Precision double casts speedup: " << (double)doublePrecTime / (double)singlePrec2CastTime
            << " ns. RelErr: mean,med,stddev,m,M " << relErrStats.mean << "," << relErrStats.med
            << "," << relErrStats.stddev << ","<< relErrStats.m << "," << relErrStats.M <<endl;

        relErrStats = relativeErrorStats<double>(doublePrec, singlePrec1CastOutput, SIZE);
        cerr << "single Precision one cast output speedup: " << (double)doublePrecTime / (double)singlePrec1CastOutputTime
            << " ns. RelErr: mean,med,stddev,m,M " << relErrStats.mean << "," << relErrStats.med
            << "," << relErrStats.stddev << ","<< relErrStats.m << "," << relErrStats.M <<endl;

        //std::cerr << "Expf("<<singlePrec[SIZE-1]<<") float: " << singlePrecTime/1000. << " ns " << inputVal[SIZE-1]<< " with cast cost." << std::endl;
        //std::cerr << "Expf("<<singlePrec[SIZE-1]<<") float: " << singlePrecTime/1000. << " ns " << inputValSinglePrec[SIZE-1]<< " without cast cost." << std::endl;
        //std::cerr << "Exp("<< doublePrec[SIZE-1]<<") double: " << doublePrecTime/1000. << " ns " << inputVal[SIZE-1]<<std::endl
    }
    return 0;
}
