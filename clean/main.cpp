#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <fstream>

using namespace std;

vector<int> findIntersection(vector<int> &a, vector<int> &b, vector<int> &ia, vector<int> &ib) {
    
	vector<int> v1 = a;
	vector<int> v2 = b;

	std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());
 
    std::vector<int> v_intersection;
 
    std::set_intersection(v1.begin(), v1.end(),
                          v2.begin(), v2.end(),
                          std::back_inserter(v_intersection));
	std::sort(v_intersection.begin(), v_intersection.end());

	for(int i = 0; i < v_intersection.size(); i++){

		for(int j = 0; j < a.size(); j++){
			if(v_intersection[i] == a[j]){
				ia.push_back(j);
				break;
			}
		}

		for(int k = 0; k < b.size(); k++){
			if(v_intersection[i] == b[k]){
				ib.push_back(k);
				break;
			}
		}

	}

	return v_intersection;
}

int main(){
	int aNums[] = {7,2,4,6};
	int bNums[] = {1,2,0,4,6,10,6};

	//int aNums[] = {7,1,7,7,4};
	//int bNums[] = {7,0,4,4,0};

	//int aNums[] = {1,2,3,6};
	//int bNums[] = {1,2,3,4,6,10,20};

	vector<int> a;
	for(int i = 0; i < sizeof(aNums)/sizeof(aNums[0]); i++){
		a.push_back(aNums[i]);
	}
	vector<int> b;
	for(int i = 0; i < sizeof(bNums)/sizeof(bNums[0]); i++){
		b.push_back(bNums[i]);
	}

	vector<int> ia;
	vector<int> ib;
	vector<int> inter = findIntersection(a, b, ia, ib);

	for(int i = 0; i < ia.size(); i++){
		cout << ia[i]+1 << " ";
	}
	cout << endl;
	for(int i = 0; i < ib.size(); i++){
		cout << ib[i]+1 << " ";
	}

	vector<vector<int>> v2d;
	v2d.push_back(ia);
	v2d.push_back(ib);

	// store the coordinate of sift feature
    std::ofstream outfile("e:\\test.dat", std::ios::binary | std::ios::out);
    if (!outfile) { throw runtime_error("Cannot open file."); }
    auto v2dAllRowsSize = v2d.size();
    outfile.write((char *)&v2dAllRowsSize, sizeof(int));
	for(int i = 0; i < v2dAllRowsSize; i++ ){
        int vSize = v2d[i].size();
        outfile.write((char *)&vSize, sizeof(int));
        outfile.write((char *)&v2d[i], sizeof(int) * v2d[i].size());
    }
	outfile.close();

	// open the coordinate of sift feature
    ifstream ifs_v2d("e:\\test.dat", ios::binary);
    if (!ifs_v2d) { throw runtime_error("Cannot open file."); }
    int AllRowsSize = 0;
    ifs_v2d.read((char *)&AllRowsSize, sizeof(int));
    vector<vector<int>> v2dTest(AllRowsSize);
    for (int i = 0; i < AllRowsSize; i++) {
		int RowSize = 0;
        ifs_v2d.read((char *)&RowSize, sizeof(int));
        ifs_v2d.read((char *)&v2dTest[i], sizeof(int) * RowSize);
    }
    ifs_v2d.close();

	system("pause");

	return 0;
}