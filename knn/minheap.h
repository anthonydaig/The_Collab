//http://www.codeproject.com/Tips/816934/Min-Binary-Heap-Implementation-in-Cplusplus

#include "vector"
#include "Eigen/Dense"
using namespace std;

class MinHeap
{
private:
    
    void BubbleDown(int index);
    void BubbleUp(int index);
    void Heapify();

public:
    vector<double> _vector;
    MinHeap(Eigen::MatrixXd array, int length);
    MinHeap(const vector<double>& vector);
    MinHeap();

    void Insert(double newValue);
    double GetMin();
    void DeleteMin();
};