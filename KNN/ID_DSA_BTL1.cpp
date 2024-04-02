#include "ID_DSA_BTL1.hpp"

int NodeCounter = 0;
int ImageCounter = 0;
int DataCounter = 0;
int kNNCounter = 0;

//**********************************************************************
//* class Image inheritance List
//**********************************************************************
template<typename T>
Image<T>::Image() {
    head = tail = nullptr;
    size = 0;
    ImageCounter++;
}
template<typename T>
Image<T>::~Image() {
    this->clear();
    ImageCounter--;
}

template <typename T>
void Image<T>::push_back(T value) {
    Node *newNode = new Node(value);
    if (this->length() == 0) {
        this->head = this->tail = newNode;
    } else {
        this->tail->next = newNode;
        this->tail = newNode;
    }
    this->size++;
}

template <typename T>
void Image<T>::push_front(T value) {
    Node *newNode = new Node(value);
    if (this->length() == 0) {
        this->head = this->tail = newNode;
    } else {
        newNode->next = this->head;
        this->head = newNode;
    }
    this->size++;
}

template <typename T>
void Image<T>::insert(int index, T value) {
    if(index < 0 || index > size) return;
    //TODO: implement task 1
    if (index == 0) { // Insert at the head of linked list
        push_front(value);
    } else if (index == this->size) { //Insert at the end of the list
        push_back(value);
    } else { // Traverse the pointer to the desired position
        Node *newNode = new Node(value);
        Node *tmp = head;
        int idx = 0;
        while (idx < index - 1) {
            tmp = tmp->next;
            idx++;
        }
        newNode->next = tmp->next;
        tmp->next = newNode;
        this->size++;
    }
}

template <typename T>
void Image<T>::remove(int index) {
    if(index < 0 || index >= size) return;
    Node *deleteNode = nullptr;
    /* Case: List have 1 element*/
    if (this->size == 1) {
        deleteNode = head;
        head = tail = nullptr;
    } else if (index == 0) { // position: 0
        deleteNode = head;
        head = head->next;
    } else if (index == this->size - 1) { // position: index - 1
        int idx = 0;
        Node *tmp = head;
        while (idx < index - 1 && tmp != nullptr) {
            tmp = tmp->next;
            idx ++;
        }
        deleteNode = tmp->next;
        tmp->next = nullptr;
        tail = tmp;
    } else {
        Node* tmp = head;
        int idx = 0;
        while (idx < index - 1 && tmp != nullptr) {
            tmp = tmp->next;
            idx ++;
        }
        deleteNode = tmp->next;
        tmp->next = deleteNode->next;
    }
    delete deleteNode;
    deleteNode = nullptr;
    size--;
}

template <typename T>
T &Image<T>::get(int index) const {
    if(index < 0 || index >= this->size)  throw std::out_of_range("get(): Out of range");
    //TODO: implement task 1
    if (this->size == 1) {
        return head->pointer;
    } else {
        Node *ptr = head;
        int idx = 0;
        while (idx < index) {
            ptr = ptr->next;
            idx++;
        }
        return ptr->pointer;
    }
}

template <typename T>
int Image<T>::length() const {
    return this->size;
}

template <typename T>
void Image<T>::clear() {
    while (head) {
        Node *tmp = head;
        head = tmp->next;
        delete tmp;
    }
    tail = nullptr;
    size = 0;
}

template <typename T>
void Image<T>::print() const {
    if(size == 0) {
        cout << "";
    }
    else {
        Node* temp = head;
        for(int i = 0; i < this->size; i++)
        {
            if(i == this->size - 1) cout << temp->pointer;
            else cout << temp->pointer << " ";
            temp = temp->next;
        }
    }
}

template <typename T>
void Image<T>::printStartToEnd(int start, int end) const {
    Node* temp = head;
    for(int i = 0; i < start; i++) temp = temp->next;
    for(int i = start; i < end && i < this->size; i++)
    {
        if(i == end - 1 || i == this->size - 1) cout << temp->pointer;
        else cout << temp->pointer << " ";
        temp = temp->next;
    }
}

template <typename T>
void Image<T>::printNoNewLine(int start, int end) const {
    Node *tmp = head;
    for (int i = start; i < end && i < this-> size; i++) {
        if (i == end - 1 || i == this->size - 1) cout << tmp->pointer;
        else cout << tmp->pointer << " ";
        tmp = tmp->next;
    }
}

template<typename T>
void Image<T>::reverse() {
    if (head == nullptr) {
        return;
    }
    // Create a new linked list
    auto *ll = new Image();
    Node *tmp = head;
    while(tmp) {
        ll->push_front(tmp->pointer);
        tmp = tmp->next;
    }
    head = ll->head;
    tail = ll->tail;
}
template<typename T>
List<T> *Image<T>::subList(int start, int end) {
    if(this->size <= start) return nullptr;
    if (end < start) return nullptr;
    //TODO: implement task 1
    List<T>* result = new Image<T>();
    // Copy all the list
    if (start < 0 && end > this->size) {
        Node *tmp = head;
        while (tmp) {
            result->push_back(tmp->pointer);
            tmp = tmp->next;
        }
    } else if (start >= 0 && end > this->size) {
        Node *ptr = head;
        for (int i = 0; i < start; i++) {
            ptr = ptr->next;
        }
        while (ptr) {
            result->push_back(ptr->pointer);
            ptr = ptr->next;
        }
    } else {
        Node *ptr = head;
        for (int i = 0; i < start; i++) {
            ptr = ptr->next;
        }
        for (int i = start; i < end; i++) {
            result->push_back(ptr->pointer);
            ptr = ptr->next;
        }
    }
    return result;
}

template<typename T>
void Image<T>::getArray(T *array) const {
    if (this->size == 0) {
        return;
    }
    Node *tmp = head;
    int idx = 0;
    while (tmp) {
        *(array + idx) = tmp->pointer;
        idx++;
        tmp = tmp->next;
    }
}

//**********************************************************************
//* class Dataset
//**********************************************************************
Dataset::Dataset() {
    this->nameCol = new Image<string>();
    this->data = new Image<List<int>*>();
    DataCounter++;
}

Dataset::~Dataset () {
    delete data;
    delete nameCol;
    DataCounter--;
}

Dataset::Dataset(const Dataset &other) {
    this->nameCol = new Image<string>();
    this->data = new Image<List<int>*>();
    // Copy all element of other.nameCol to this->nameCol
    int totalCols = other.nameCol->length();
    for (int i = 0; i < totalCols; i++) {
        this->nameCol->push_back(other.nameCol->get(i));
    }
    int totalRows = other.data->length();
    for (int r = 0; r < totalRows; r++) {
        this->data->push_back(other.data->get(r));
    }
}

Dataset &Dataset::operator=(const Dataset &other) {
    this->nameCol = new Image<string>();
    this->data = new Image<List<int>*>();
    // Copy all element of other.nameCol to this->nameCol
    int totalCols = other.nameCol->length();
    for (int i = 0; i < totalCols; i++) {
        this->nameCol->push_back(other.nameCol->get(i));
    }
    int totalRows = other.data->length();
    for (int r = 0; r < totalRows; r++) {
        this->data->push_back(other.data->get(r));
    }
    return *this;
}

bool Dataset::loadFromCSV(const char *fileName) {
    ifstream file(fileName);
    //* kiểm tra mở file
    if(file.is_open())
    {
        string str;
        int number;
        file >> str;
        for (int i = 0; i < str.length(); i++) {
            if (str[i] == ',') str[i] = ' ';
        }
        stringstream ss(str);
        while(ss >> str) nameCol->push_back(str);
        while(file >> str)
        {
            for (int i = 0; i < str.length(); i++) {
                if (str[i] == ',') str[i] = ' ';
            }
            stringstream ss(str);
            List<int>* temp = new Image<int>();
            while(ss >> number) temp->push_back(number);
            data->push_back(temp);
        }
        return true;
    }
    return false;
}

void Dataset::getShape(int &nRows, int &nCols) const {
    if (this->data->length() == 0) {
        nRows = 0;
        nCols = 0;
        return;
    }
    nRows = this->data->length();
    nCols = this->data->get(0)->length();
}

void Dataset::columns() const {
    int totalCols = this->nameCol->length();
    this->nameCol->printStartToEnd(0, totalCols);
    cout << endl;
}

void Dataset::rows() const {
    int totalRows = this->data->length();
    if (totalRows == 0) return;
    int totalDataCols = this->data->get(0)->length();
    if (totalDataCols == 0) return;
    for (int i = 0; i < totalRows; i++) {
        if (i == totalRows - 1) {
            this->data->get(i)->printStartToEnd(0, totalDataCols);
        } else {
            this->data->get(i)->printStartToEnd(0, totalDataCols);
            cout << endl;
        }

    }
}

void Dataset::printHead(int nRows, int nCols) const {
    if(nRows <= 0 || nCols <= 0) return;
    /* Print name column
     * even if data is none
     * */
    int totalNameCol = this->nameCol->length();
    int totalRows = this->data->length();
    if (totalNameCol <= 0) {
        if (totalRows <= 0) {
            return;
        } else {
            this->rows();
        }
    } else {
        if (nRows >= totalRows && nCols >= totalNameCol) {
            this->columns();
            this->rows();
        } else {
            if (nCols > totalNameCol) nCols = totalNameCol;
            this->nameCol->printStartToEnd(0, nCols);
            cout << endl;
            if (totalRows <= 0) {
                return;
            } else {
                int totalDataCols = this->data->get(0)->length();
                if (nRows >= totalRows) nRows = totalRows;
                if (nCols >= totalDataCols) nCols = totalDataCols;
                for (int i = 0; i < nRows; i++) {
                    if (i == nRows -  1) {
                        this->data->get(i)->printStartToEnd(0, nCols);
                    } else {
                        this->data->get(i)->printStartToEnd(0, nCols);
                        cout << endl;
                    }
                }
                // this->data->get(nRows - 1)->printNoNewLine(0, nCols);
            }
        }
    }
}

void Dataset::printTail(int nRows, int nCols) const {
    if(nRows <= 0 || nCols <= 0)  return;
    /* Print name column
     * even if data is none
     * */
    int totalNameCol = this->nameCol->length();
    int totalRows = this->data->length();
    if (totalNameCol <= 0) {
        if (totalRows <= 0) {
            return;
        } else {
            this->rows();
        }
    } else {
        if (nRows >= totalRows && nCols >= totalNameCol) {
            this->columns();
            this->rows();
        } else {
            if (nCols > totalNameCol) nCols = totalNameCol;
            this->nameCol->printStartToEnd(totalNameCol - nCols, totalNameCol);
            cout << endl;
            if (totalRows <= 0) {
                return;
            } else {
                int totalDataCols = this->data->get(0)->length();
                if (nRows >= totalRows) nRows = totalRows;
                if (nCols >= totalDataCols) nCols = totalDataCols;
                if (totalDataCols <= 0) {
                    return;
                }
                for (int i = 0; i < nRows; i++) {
                    if (i == nRows - 1) {
                        this->data->get(totalRows - nRows + i)->printStartToEnd(totalDataCols - nCols, totalDataCols);
                    } else {
                        this->data->get(totalRows - nRows + i)->printStartToEnd(totalDataCols - nCols, totalDataCols);
                        cout << endl;
                    }

                }
                // this->data->get(totalRows - nRows + nRows - 1)->printNoNewLine(totalDataCols - nCols, totalDataCols);
            }
        }
    }
}

int Dataset::getColumnIndex(std::string columns) const {
    int totalColumns = this->nameCol->length();
    for (int c = 0; c < totalColumns; c++) {
        if (columns.compare(this->nameCol->get(c)) == 0) {
            return c;
        }
    }
    return -1;
}

bool Dataset::drop(int axis, int index, std::string columns) {
    if (this->nameCol->length() == 0) return false;
    if (this->data->length() == 0) {
        if (axis == 1) {
            int foundColumn = this->getColumnIndex(columns);
            if (foundColumn == -1) return false;
            this->nameCol->remove(foundColumn);
            return true;
        } else if (axis == 0) {
            if (index < 0 || index >= this->data->length()) return false;
            return true;
        } else {
            return false;
        }
    } else if (axis == 1) {
        int foundColumn = this->getColumnIndex(columns);
        if (foundColumn == -1) return false;
        this->nameCol->remove(foundColumn);
        for (int i = 0; i < this->data->length(); i++) {
            this->data->get(i)->remove(foundColumn);
        }
        return true;
    } else if (axis == 0) {
        if (index < 0 || index >= this->data->length()) return false;
        this->data->remove(index);
        return true;
    } else {
        return false;
    }
}

Dataset Dataset::extract(int startRow, int endRow, int startCol, int endCol) const {
    Dataset result;
    Dataset *ptr = &result;
    int totalRows = this->data->length();
    int totalCols = this->nameCol->length();
    int dataCols = totalCols;
    if (totalRows == 0) throw std::out_of_range("get(): Out of range"); // If total row = 0 throw std::out_of_range("get(): Out of range");
    if (totalCols < this->data->get(0)->length()) {
        dataCols = this->data->get(0)->length();
    }
    if ((startCol >= dataCols) || (startRow >= totalRows)
        || (startCol < 0) || (startRow < 0))  {
        throw std::out_of_range("get(): Out of range");
    }

    if (endCol == -1 || endCol >= totalCols) {
        endCol = dataCols - 1;
        for (int i = startCol; i < totalCols; i++) {
            ptr->nameCol->push_back(this->nameCol->get(i));
        }
    } else if(endCol < startCol) {
        throw std::out_of_range("get(): Out of range");
    } else {
        for (int c = startCol; c <= endCol; c++) {
            ptr->nameCol->push_back(this->nameCol->get(c));
        }
    }

    if (endRow == -1 || endRow >= totalRows) {
        endRow = totalRows - 1;
    } else {
        if (endRow < startRow) throw std::out_of_range("get(): Out of range");
    }
    for (int r = startRow; r < endRow + 1; r++) {
        ptr->data->push_back(this->data->get(r)->subList(startCol, endCol + 1));
    }
    return *ptr;
}

List<List<int> *> *Dataset::getData() const {
    return this->data;
}

double Dataset::distanceEuclidean(const List<int> *x, const List<int> *y) const {
    int x_len = x->length();
    int y_len = y->length();
    if (x_len <= 0 && y_len <= 0) {
        return 0.0;
    }

    double distance = 0.0;
    if (x_len == 0 && y_len > 0) {
        int *Y = new int[y_len];
        y->getArray(Y);
        for (int i = 0; i < y_len; i++) {
            distance += pow(Y[i], 2);
        }
        return sqrt(distance);
    }
    if (x_len > 0 && y_len == 0) {
        int *X = new int[x_len];
        x->getArray(X);
        for (int i = 0; i < x_len; i++) {
            distance += pow(X[i], 2);
        }
        return sqrt(distance);
    }

    int tmpIdx = 0;

    int* X = new int[x_len];
    int* Y = new int[y_len];
    x->getArray(X);
    y->getArray(Y);

    if (x_len == 1 && y_len == 1) {
        distance += pow((double)X[0] -(double)Y[0], 2);
        return sqrt(distance);
    }
    if (x_len == y_len) {
        for (int i = 0; i < x_len; i++) {
            distance += pow((double)X[i] - (double)Y[i], 2);
        }
        return sqrt(distance);
    }
    for (int i = 0; i < x_len; i++) {
        if (tmpIdx == y_len - 1) {
            distance += pow((double)X[i], 2);
        } else {
            distance += pow((double)X[i] - (double)Y[i], 2);
        }
        tmpIdx++;
    }
    while (tmpIdx < y_len) {
        distance += pow((double)Y[tmpIdx], 2);
        tmpIdx++;
    }

    delete[] X;
    delete[] Y;
    return sqrt(distance);
}

void merge(int *label, double *distances, int start, int mid, int end) {
    // Copy label and distances to temp left and right array
    int leftElements = mid - start + 1;
    int rightElements = end - mid;

    /*Left array*/
    int *leftLabel = new int[leftElements];
    double *leftDistance = new double[leftElements];
    /*Right array*/
    int *rightLabel = new int[rightElements];
    double *rightDistance = new double[rightElements];

    for (int i = 0; i < leftElements; i++) {
        leftLabel[i] = label[start + i];
        leftDistance[i] = distances[start + i];
    }

    for (int i = 0; i < rightElements; i++) {
        rightLabel[i] = label[mid + 1 + i];
        rightDistance[i] = distances[mid + 1 + i];
    }

    int leftIndex = 0;
    int rightIndex = 0;
    int mergeIndex = start;

    while (leftIndex < leftElements && rightIndex < rightElements) {
        if (leftDistance[leftIndex] <= rightDistance[rightIndex]) {
            label[mergeIndex] = leftLabel[leftIndex];
            distances[mergeIndex] = leftDistance[leftIndex];
            leftIndex++;
        } else {
            label[mergeIndex] = rightLabel[rightIndex];
            distances[mergeIndex] = rightDistance[rightIndex];
            rightIndex++;
        }
        mergeIndex++;
    }

    while (leftIndex < leftElements) {
        label[mergeIndex] = leftLabel[leftIndex];
        distances[mergeIndex] = leftDistance[leftIndex];
        leftIndex++;
        mergeIndex++;
    }

    while (rightIndex < rightElements) {
        label[mergeIndex] = rightLabel[rightIndex];
        distances[mergeIndex] = rightDistance[rightIndex];
        rightIndex++;
        mergeIndex++;
    }

    delete[] leftLabel;
    delete[] leftDistance;

    delete[] rightLabel;
    delete[] rightDistance;
}

void mergeSort(int *label, double *distances, int start, int end) {
    if (start >= end)
        return;
    int mid = start + (end - start) / 2;
    mergeSort(label, distances, start, mid);
    mergeSort(label, distances, mid + 1, end);
    merge(label, distances, start, mid, end);
}

/* Bubble sort
 * Worst case: O(n^2)
 * */
void bubbleSort(double* distances, int *label, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = n - 1; j > i; --j) {
            if (distances[j] < distances[j - 1]) {
                swap(distances[j], distances[j - 1]);
                swap(label[j], label[j - 1]);
            }
        }
    }
}



int maxFreq(int freq[], int n) {
    int max = 0;
    for (int i = 1; i < n; i++) {
        if (freq[max] < freq[i]) {
            max = i;
        }
    }
    return max;
}

int assignPredict(int *label, int k, int n) {
    int targetLabel = 0;
    /* Initialize the frequency array
     * */
    int freq[10];
    for (int i = 0; i < 10; i++) {
        freq[i] = 0;
    }

    for (int i = 0; i < k; i++) {
        freq[label[i]] += 1;
    }
    targetLabel = maxFreq(freq, 10);
    return targetLabel;
}

Dataset Dataset::predict(const Dataset &X_train, const Dataset &Y_train, const int k) const {
    if (k < 1) throw std::out_of_range("get(): Out of range");
    Dataset y_predict = {};
    Dataset *ptr = &y_predict;

    /* Y_train should have shape (total_rows, 1)
     * X_train should have shape (total_rows, total_columns)
     * */
    int totalTrainRows = X_train.data->length();
    if (Y_train.nameCol->length() == 0) throw std::out_of_range("get(): Out of range"); // If Y_train has no label
    if (totalTrainRows == 0) throw std::out_of_range("get(): Out of range");            // If X_train has no data
    if (k > totalTrainRows) throw std::out_of_range("get(): Out of range");
    ptr->nameCol->push_back(Y_train.nameCol->get(0));                                   // Assign label for predict dataset

    /* Calculate distance from data point in test set
     * to data point in train set
     * total rows of test set is testRow   (below)
     * total rows of train set is trainRos (below)
     * */
    int totalTestRows = this->data->length();                       // This represent total rows in test set
    for (int testRow = 0; testRow < totalTestRows; testRow++) {
        //TODO
        /* We traverse through X_test
         * for each point in X_test, we calculate distance Fto points
         * in X_train.
         * */
        double *distances = new double[totalTrainRows + 1];         // Allocate new array to save distances
        int *label = new int[totalTrainRows + 1];                   // Allocate new array to save labels
        for (int trainRow = 0; trainRow < totalTrainRows; trainRow++) {
            *(distances + trainRow) = distanceEuclidean(this->data->get(testRow), X_train.data->get(trainRow));
            *(label + trainRow) = Y_train.data->get(trainRow)->get(0);
        }
        mergeSort(label, distances, 0, totalTrainRows - 1);
        int targetLabel = assignPredict(label, k, totalTrainRows);
        List<int> *value = new Image<int>();
        value->push_back(targetLabel);
        ptr->data->push_back(value);

        delete[] distances;
        delete[] label;
    }
    return *ptr;
}

double Dataset::score(const Dataset &y_predict) const {
    int predictedLen = y_predict.data->length();
    int labelLen = this->data->length();
    if (labelLen == 0 || predictedLen == 0) return -1.0;
    if (predictedLen != labelLen) return -1.0;
    int score = 0;
    for (int i = 0; i < labelLen; i++) {
        if (y_predict.data->get(i)->get(0) == this->data->get(i)->get(0)) {
            score++;
        }
    }
    double result = (double) score / (double)labelLen;
    return result;
}

//**********************************************************************
//* Class kNN
//**********************************************************************
kNN::kNN(int k) {
    this->k = k;
}

Dataset kNN::predict(const Dataset &X_test) {
    return X_test.predict(this->X_train, this->Y_train, this->k);
}

double kNN::score(const Dataset &y_test, const Dataset &y_pred) {
    return y_test.score(y_pred);
}

void kNN::fit(const Dataset &X_train, const Dataset &y_train) {
    this->X_train = X_train;
    this->Y_train = y_train;
}

//**********************************************************************
//* hàm train_test_split
//**********************************************************************
void train_test_split(Dataset &X, Dataset &Y, double test_size,
                      Dataset &X_train, Dataset &X_test, Dataset &Y_train, Dataset &Y_test)
{
    if (X.getData()->length() != Y.getData()->length() || test_size >= 1 || test_size <= 0)
        return;

    double minDouble = 1.0e-15;
    int nRow = X.getData()->length();
    double rowSplit = nRow * (1 - test_size);

    if (abs(round(rowSplit) - rowSplit) < minDouble * nRow)
        rowSplit = round(rowSplit);

    X_train = X.extract(0, rowSplit - 1, 0, -1);
    Y_train = Y.extract(0, rowSplit - 1, 0, -1);

    X_test = X.extract(rowSplit, -1, 0, -1);
    Y_test = Y.extract(rowSplit, -1, 0, -1);
}