//http://www.codeproject.com/Tips/816934/Min-Binary-Heap-Implementation-in-Cplusplus

#include "vector"
using namespace std;

class MinHeap
{
private:
    vector<int> _vector;
    void BubbleDown(int index);
    void BubbleUp(int index);
    void Heapify();

public:
    MinHeap(int* array, int length);
    MinHeap(const vector<int>& vector);
    MinHeap();

    void Insert(int newValue);
    int GetMin();
    void DeleteMin();
};