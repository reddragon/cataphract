#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstring>
#include <chrono>

using namespace std;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::steady_clock;

class DTNode {
  public:
  bool isLeaf;
  float label;
  int nodes;
  DTNode *left, *right;
  int splitFeat;
  float splitValue;
};

int leafNodes = 0;

struct DTData {
  public:
  // Choosing the raw pointer instead of any smart pointers,
  // because I saw a ~10x performance drop with smart
  // pointers. To be fair, I didn't experiment much, I could
  // easily manually clean up the allocs, and I went with that.
  // Don't judge me, this is not how I code for prod :D
  vector<float>* values;
  float label;
};

void writeTree(DTNode *tree, ofstream& ofs) {
  ofs << tree->isLeaf << endl;
  if (tree->isLeaf) {
    ofs << tree->label << endl;
    ofs << tree->nodes << endl;
  } else {
    ofs << tree->splitFeat << endl;
    ofs << tree->splitValue << endl;
    ofs << tree->nodes << endl;
    writeTree(tree->left, ofs);
    writeTree(tree->right, ofs);
  }
} 

DTNode* readTree(ifstream& ifs) {
  DTNode* n = new DTNode();
  ifs >> n->isLeaf;
  if (n->isLeaf) {
    ifs >> n->label;  
    ifs >> n->nodes;
  } else {
    ifs >> n->splitFeat;
    ifs >> n->splitValue;
    ifs >> n->nodes;
    n->left = readTree(ifs);
    n->right = readTree(ifs);
  }
  return n;
}

double entropy(int p, int n, int total) {
  if (p == 0 || n == 0) {
    return 0.0;
  }
  return -(p * log(p * 1.0/ total) + n * log(n * 1.0 /total));
}

DTNode* learn(vector<DTData>& data, int start, int end) {
  int total = end - start;
  int p = 0, n = 0;
  for (int i = start; i < end; i++) {
    if (data[i].label == 1.0) {
      p++;
    } else {
      n++;
    }
  }
  
  // TODO
  // Change this to reflect when we are not going to explore 
  // any further.
  double initEntropy = entropy(p, n, total);
  // cout << "Entropy: " << initEntropy << ", size: " << total << endl;
  double thresholdEntropy = 0.0;
  if (initEntropy < thresholdEntropy) {
    DTNode* dn = new DTNode();
    dn->isLeaf = true;
    leafNodes++;
    dn->label = (p * 1.0 / total);
    // cout << "Storing " << dn->label << ", p: " << p << ", n: " << n << ", total: " << total << ", entropy: " << initEntropy << endl;
    dn->nodes = 1;
    return dn;
  }

  int totalFeats = data[start].values->size();

  double gain = -1;
  int bestFeat = -1;
  float splitVal = -1.0;
  int splitIndex = -1;

  for (int f = 0; f < totalFeats; f++) {
    sort(data.begin() + start, data.begin() + end, [=] (DTData x, DTData y) { return (*x.values)[f] < (*y.values)[f]; } );
    
    int pp = 0, nn = 0, ttotal = 0;
    double lBestGain = 0;
    float lSplitVal = -1.0;
    int lSplitIndex = -1;

    for (int i = start; i < end; i++) {
      ttotal++;
      if (data[i].label == 1.0) {
        pp++;
      } else {
        nn++;
      }

      if (i != end - 1 && (*data[i].values)[f] != (*data[i+1].values)[f]) {
        double curGain = initEntropy - (entropy(pp, nn, ttotal) + entropy(p - pp, n - nn, total - ttotal));
        if (curGain > lBestGain) {
          lBestGain = curGain;
          lSplitVal = (*data[i].values)[f];
          lSplitIndex = i;
        }
      }
    }

    if (lBestGain > gain) {
      gain = lBestGain;
      splitVal = lSplitVal;
      bestFeat = f;
      splitIndex = lSplitIndex;
    }
  }
  
  double gainCutoff = 10;
  if (splitIndex == -1 || gain <= gainCutoff) {
    // cout << "Could not find a split" << endl;
    DTNode* dn = new DTNode();
    dn->isLeaf = true;
    leafNodes++;
    dn->label = (p * 1.0 / total);
    dn->nodes = 1;
    return dn;
  }

  sort(data.begin() + start, data.begin() + end, [=] (DTData x, DTData y) { return (*x.values)[bestFeat] < (*y.values)[bestFeat]; } );
  DTNode *dn = new DTNode();
  dn->isLeaf = false;
  dn->splitFeat = bestFeat;
  dn->splitValue = splitVal;
  dn->left = learn(data, start, splitIndex + 1);
  dn->right = learn(data, splitIndex + 1, end);
  dn->nodes = dn->left->nodes + dn->right->nodes + 1;
  return dn;
}

float evaluate(DTData d, DTNode *dn) {
  if (dn->isLeaf) {
    //cout << "Returning " << dn->label << endl;
    return dn->label;
  }
  
  if ((*d.values)[dn->splitFeat] <= dn->splitValue) {
    return evaluate(d, dn->left);
  } 
  return evaluate(d, dn->right);
} 


// This method might need to be changed for other data.
// I am reading lines in the format:
//   <label> <values (in char format)>
DTData parseLine(string line) {
  DTData ret;
  ret.values = new vector<float>();
  stringstream ss(line);
  ss >> ret.label;
  
  char f;
  while (ss >> f) {
    ret.values->push_back((float)f);
  }
 
  return ret;
}

/**
 * Gets the area under the ROC curve.
 *
 * I had to search a lot to find out how this is calculated.
 * ROC is basically a fantastic measure
 */
double rocAUC(vector< pair<bool, float> >& e) {
  sort(e.begin(), e.end(), 
       [] (pair<bool, float> x, pair<bool, float> y) {
         return x.second < y.second; 
       });
  
  long N = e.size();
  long p = 0, n = 0;
  for (long i = 0; i < e.size(); i++) {
    if (e[i].first) {
      p++;
    } else {
      n++;
    }
  }
  
  if (p == 0 || p == N) {
    return 1.0;
  }

  double prevTPR = 0, prevFPR = 0;
  double newTPR = 1, newFPR = 1;
  long pSeen = 0, nSeen = 0;
  double area = 0;
  double cutPoint = e[0].second;
  long cuts = 0;
  for (long i = 0; i < e.size(); i++) {
    if (e[i].second != cutPoint) {

      // Set the cutPoint to the current threshold.
      cutPoint = e[i].second;

      prevTPR = newTPR;
      prevFPR = newFPR;
        
      // All values less than this cutPoint are false,
      // all above this are true.
      newFPR = (n - nSeen) * 1.0 / (n);
      newTPR = (p - pSeen) * 1.0 / (p);
      
      double xIntercept = prevFPR - newFPR;
      double delta = xIntercept * (newTPR + prevTPR);
      area += delta;
    }
    
    if (e[i].first) {
      pSeen++;
    } else {
      nSeen++;
    }
  }
  
  prevTPR = newTPR;
  prevFPR = newFPR;

  // All values less than this cutPoint are false,
  // all above this are true.
  newFPR = (n - nSeen) * 1.0 / (n);
  newTPR = (p - pSeen) * 1.0 / (p);

  double xIntercept = prevFPR - newFPR;
  double delta = xIntercept * (newFPR + newTPR);
  area += delta;

  area /= 2.0;
  return area;
}

/**
 * Evaluates the decision tree that we have built, on the input.
 * Metrics are:
 * - MSE: 
 *   Mean Squared Error (Less is better)
 * - TM Accuracy: 
 *   Accuracy, as defined in Tom Mitchell's ML course. (More is better)
 * - ROC: 
 *   ~ 0.5 is random behavior
 *   ~ 0.6 on test is overfitting 
 *   >= 0.8 on test is good
 */
void checkPerformance(vector<DTData>& data, DTNode* head) {
  double rmsError = 0;
  double accuracy = 0.0;
  int falsePos = 0, falseNeg = 0;
  
  vector< pair<bool, float> > p;
  for (int i = 0; i < data.size(); i++) {
    float answer = evaluate(data[i], head);
    rmsError += (answer - data[i].label) * (answer - data[i].label);
    bool binaryPred = (answer > 0.5);
    bool binaryLabel = (data[i].label == 1.0);
    p.push_back(make_pair(binaryLabel, answer));
    if (binaryLabel == true && binaryPred == false) {
      falseNeg++;
    } else if (binaryLabel == false && binaryPred == true) {
      falsePos++;
    }
  }
  
  
  rmsError /= (data.size() * 1.0);
  accuracy = (data.size() - falsePos - falseNeg) * 1.0 / (data.size());
  cout << "MS Error: " << rmsError << endl; 
  cout << "TM Accuracy: " << accuracy << endl;
  cout << "ROC: " << rocAUC(p) << endl;
}

void cleanupDTNode(DTNode* head) {
  if (head == NULL) {
    return;
  }
  cleanupDTNode(head->left);
  cleanupDTNode(head->right);
  delete head;
}

void cleanup(vector<DTData>& data, DTNode* head) {
  for (int i = 0; i < data.size(); i++) {
    delete data[i].values;
  }
  cleanupDTNode(head);
}

void train(char* fileName) {
  vector<DTData> data;
  ifstream ifs;
  ifs.open(fileName);
  
  string line;
  cout << "Reading " << fileName << endl;
  while (getline(ifs, line)) {
    data.push_back(parseLine(line));
  }
  cout << "Done with reading the data" << endl;
  ifs.close();
  
  cout << "Starting to learn." << endl;
  leafNodes = 0;
  steady_clock::time_point startTime = steady_clock::now();
  DTNode* head = learn(data, 0, data.size());
  steady_clock::time_point endTime = steady_clock::now();
  cout << "Done with learning. " << head->nodes << " nodes with " << leafNodes << " leaves." << endl;
  cout << "Time taken for learning: "
       << (duration_cast<milliseconds>(endTime - startTime).count()) / 1000.0
       << "s." << endl;

  cout << "Writing the tree to disk." << endl;
  ofstream ofs;
  ofs.open("tree.out");
  writeTree(head, ofs);
  ofs.close();
  cout << "Done with writing the tree to disk." << endl;
  
  cout << "Evaluating on the training data." << endl;
  checkPerformance(data, head);

  cout << "Manually cleaning up the allocs." << endl;
  startTime = steady_clock::now();
  cleanup(data, head);
  endTime = steady_clock::now();
  cout << "Time taken for cleaning up: "
       << (duration_cast<milliseconds>(endTime - startTime).count())
       << "ms." << endl;
}


void test(char* fileName) {
  vector<DTData> data;
  ifstream ifs;
  ifs.open(fileName);
  
  string line;
  cout << "Reading " << fileName << endl;
  while (getline(ifs, line)) {
    data.push_back(parseLine(line));
  }
  cout << "Done with reading the data" << endl;
  ifs.close();
  
  cout << "Starting to read tree from disk." << endl;
  ifstream tf;
  tf.open("tree.out");
  DTNode* head = readTree(tf);
  tf.close();
  cout << "Done with reading." << endl;

  cout << "Evaluating on the input data." << endl;
  checkPerformance(data, head);
  
  cout << "Manually cleaning up the allocs." << endl;
  steady_clock::time_point startTime = steady_clock::now();
  cleanup(data, head);
  steady_clock::time_point endTime = steady_clock::now();
  cout << "Time taken for cleaning up: "
       << (duration_cast<milliseconds>(endTime - startTime).count())
       << "ms." << endl;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    cout << "Incorrect usage" << endl;
    return 1;
  }
  if (!strcmp(argv[1], "train")) {
    train(argv[2]); 
  } else if (!strcmp(argv[1], "test")) {
    test(argv[2]);  
  } else {
    cout << "Incorrect usage" << endl;
    return 1;
  }
  return 0;
}
