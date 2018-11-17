#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
	if(argc != 4)
	{
		cerr << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
		return 1;
	}

	// Retrieve paths to images
	vector<string> vstrImageFilenames;
	vector<double> vTimestamps;
	string strFile = string(argv[3])+"/rgb.txt";
	LoadImages(strFile, vstrImageFilenames, vTimestamps);

	int nImages = vstrImageFilenames.size();

	// Create SLAM system. It initializes all system threads and gets ready to process frames.
	MySLAM::System SLAM(argv[1],argv[2],MySLAM::System::MONOCULAR,true);
	
	// Vector for tracking time statistics
	vector<float> vTimesTrack;
	vTimesTrack.resize(nImages);

	cout << endl << "-------" << endl;
	cout << "Start processing sequence ..." << endl;
	cout << "Images in the sequence: " << nImages << endl << endl;

	// Main loop
	cv::Mat im;
	for(int ni=0; ni<nImages; ni++)
	{
		// Read image from file
		im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],cv::IMREAD_UNCHANGED);
		double tframe = vTimestamps[ni];

		if(im.empty())
		{
			cerr << endl << "Failed to load image at: "
				 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
			return 1;
		}

		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

		// Pass the image to the SLAM system
		SLAM.Track(im,tframe);

		std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

		double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
		vTimesTrack[ni]=ttrack;

		// Wait to load the next frame
		double T=0;
		if(ni<nImages-1)
			T = vTimestamps[ni+1]-tframe;
		else if(ni>0)
			T = tframe-vTimestamps[ni-1];

		if(ttrack<T)
			this_thread::sleep_for(chrono::microseconds(int((T - ttrack)*1e6)));
	}
	
	cout << "Press ENTER to quit ." << endl;
	getchar();

	// Stop all threads
	SLAM.Shutdown();

	return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
	ifstream f;
	f.open(strFile.c_str());

	while(!f.eof())
	{
		string s;
		getline(f,s);
		if(!s.empty())
		{
			stringstream ss;
			ss << s;
			double t;
			string sRGB;
			ss >> t;
			vTimestamps.push_back(t);
			ss >> sRGB;
			vstrImageFilenames.push_back(std::move(sRGB));
		}
	}
}
