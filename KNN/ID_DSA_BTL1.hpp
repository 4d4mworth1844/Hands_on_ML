#include "main.hpp"

extern int NodeCounter;
extern int ImageCounter;
extern int DataCounter;
extern int kNNCounter;

template <typename T>
class List
{
public:
    virtual ~List() = default;
    virtual void push_back(T value) = 0;
    virtual void push_front(T value) = 0;
    virtual void insert(int index, T value) = 0;
    virtual void remove(int index) = 0;
    virtual T &get(int index) const = 0;
    virtual int length() const = 0;
    virtual void clear() = 0;
    virtual void print() const = 0;
    virtual void reverse() = 0;

    //! Hàm hỗ trợ thêm
    virtual List<T> *subList(int start, int end) = 0;
    virtual void printStartToEnd(int start, int end) const = 0;
    // ! Hàm bắt buộc hiện thục để giảm độ phức tạp
    // TODO: Chép các giá trị pointer vào mảng
    virtual void getArray(T *array) const = 0;
    virtual void printNoNewLine(int start, int end) const = 0;
};

// class Image
template <typename T>
class Image : public List<T>
{
public:
    class Node
    {
    public:
        T pointer;
        Node *next;
        Node(T pointer, Node *next = nullptr) : pointer(pointer), next(next) {}
    };

private:
    Node *head;
    Node *tail;
    int size;

public:
    Image();
    ~Image();
    void push_back(T value);
    void push_front(T value);
    void insert(int index, T value);
    void remove(int index);
    T &get(int index) const;
    int length() const;
    void clear();
    void print() const;
    void reverse();
    void printStartToEnd(int start, int end) const;
    List<T> *subList(int start, int end);
    void getArray(T *array) const;
    void printNoNewLine(int start, int end) const;
};

class Dataset
{
private:
    List<List<int> *> *data;
    List<string> *nameCol;
    // You may need to define more
public:
    Dataset();
    ~Dataset();
    Dataset(const Dataset &other);
    Dataset &operator=(const Dataset &other);
    bool loadFromCSV(const char *fileName);
    void printHead(int nRows = 5, int nCols = 5) const;
    void printTail(int nRows = 5, int nCols = 5) const;
    void getShape(int &nRows, int &nCols) const;
    void columns() const;
    void rows() const;
    int getColumnIndex(std::string columns) const;
    bool drop(int axis = 0, int index = 0, std::string columns = "");
    Dataset extract(int startRow = 0, int endRow = -1, int startCol = 0, int endCol = -1) const;
    List<List<int> *> *getData() const;

    //! Hàm hỗ trợ thêm
    // TODO: Gọi về getArray để xủ lý giảm độ phức tạp khi truy xuất phần tử
    double distanceEuclidean(const List<int> *x, const List<int> *y) const;
    Dataset predict(const Dataset &X_train, const Dataset &Y_train, const int k) const;
    double score(const Dataset &y_predict) const;
};

class kNN
{
private:
    int k;
    Dataset X_train;
    Dataset Y_train;
    // You may need to define more
public:
    kNN(int k = 5);
    void fit(const Dataset &X_train, const Dataset &y_train);
    Dataset predict(const Dataset &X_test);
    double score(const Dataset &y_test, const Dataset &y_pred);
};
// Please add more or modify as needed
void train_test_split(Dataset &X, Dataset &Y, double test_size,
                      Dataset &X_train, Dataset &X_test, Dataset &Y_train, Dataset &Y_test);