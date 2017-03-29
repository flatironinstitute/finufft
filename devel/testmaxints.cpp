#include <iostream>
using namespace std;

int main(int argc, char* argv[])
// test integer overflow, size of long
// Barnett 3/28/17
// compile and run:
// g++ testmaxints.cpp -o testmaxints; ./testmaxints
{
  long long s=1000000000;  // step is 1e9
  for (long n=0; n<=10*s; n+=s)      // stops at 5e9, proves long = int64
    cout << n << "\n";               // else runs forever if long = int32

  // check some ptr sizes...
  double a; cout << sizeof(&a) << "\n";
  float b; cout << sizeof(&b) << "\n";
  
  return 0;
}
