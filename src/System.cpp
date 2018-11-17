#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>

bool has_suffix(const std::string &str, const std::string &suffix)
{
	std::size_t index = str.find(suffix, str.size() - suffix.size());
	return (index != std::string::npos);
}

namespace MySLAM
{

System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
			   const bool bUseViewer):mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false),mbActivateLocalizationMode(false),
		mbDeactivateLocalizationMode(false)
{
	cout << "Monocular SLAM ." << endl;

	//Check settings file
	cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
	if(!fsSettings.isOpened())
	{
	   cerr << "Failed to open settings file at: " << strSettingsFile << endl;
	   exit(-1);
	}


	//Load ORB Vocabulary
	cout << endl << "Loading ORB Vocabulary..." << endl;

	mpVocabulary = new ORBVocabulary();
	bool bVocLoad = false; // chose loading method based on file extension
    if (has_suffix(strVocFile, ".txt"))
		bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
	else
		bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);
	if(!bVocLoad)
	{
		cerr << "Wrong path to vocabulary. " << endl;
		cerr << "Falied to open at: " << strVocFile << endl;
		exit(-1);
	}
	cout << "Vocabulary loaded!" << endl << endl;

	//Create KeyFrame Database
	mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

	//Create the Map
	mpMap = new Map();

	//Create Drawers. These are used by the Viewer
	mpFrameDrawer = new FrameDrawer(mpMap);
	mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

	//Initialize the Tracking thread
	//(it will live in the main thread of execution, the one that called this constructor)Tracking线程在main主线程中
	mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer, mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

	//Initialize the Local Mapping thread and launch
	mpLocalMapper = new LocalMapping(mpMap, true);
	mptLocalMapping = new thread(&MySLAM::LocalMapping::Run,mpLocalMapper);

	//Initialize the Loop Closing thread and launch
	mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, false);
	mptLoopClosing = new thread(&MySLAM::LoopClosing::Run, mpLoopCloser);

	//Initialize the Viewer thread and launch
	if(bUseViewer)
	{
		mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
		mptViewer = new thread(&Viewer::Run, mpViewer);
		mpTracker->SetViewer(mpViewer);
	}

	//Set pointers between threads
	mpTracker->SetLocalMapper(mpLocalMapper);
	mpTracker->SetLoopClosing(mpLoopCloser);

	mpLocalMapper->SetTracker(mpTracker);
	mpLocalMapper->SetLoopCloser(mpLoopCloser);

	mpLoopCloser->SetTracker(mpTracker);
	mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

cv::Mat System::Track(const cv::Mat &im, const double &timestamp)
{
	// Check mode change
	{
		unique_lock<mutex> lock(mMutexMode);
		if(mbActivateLocalizationMode)
		{
			mpLocalMapper->RequestStop();

			// Wait until Local Mapping has effectively stopped
			while(!mpLocalMapper->isStopped())
			{
				//usleep(1000);
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}

			mpTracker->InformOnlyTracking(true);/* 只跟踪 */
			mbActivateLocalizationMode = false;
		}
		if(mbDeactivateLocalizationMode)
		{
			mpTracker->InformOnlyTracking(false);/* 跟踪+定位 */
			mpLocalMapper->Release();
			mbDeactivateLocalizationMode = false;
		}
	}

	// Check reset
	{
		unique_lock<mutex> lock(mMutexReset);
		if(mbReset)
		{
			mpTracker->Reset();
			mbReset = false;
		}
	}

	cv::Mat Tcw = mpTracker->GrabImage(im,timestamp);

	unique_lock<mutex> lock2(mMutexState);
	mTrackingState = mpTracker->mState;
	mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
	mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

	return Tcw;
}

void System::ActivateLocalizationMode()
{
	unique_lock<mutex> lock(mMutexMode);
	mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
	unique_lock<mutex> lock(mMutexMode);
	mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
	static int n=0;
	int curn = mpMap->GetLastBigChangeIdx();
	if(n<curn)
	{
		n=curn;
		return true;
	}
	else
		return false;
}

void System::Reset()
{
	unique_lock<mutex> lock(mMutexReset);
	mbReset = true;
}

void System::Shutdown()
{
	mpLocalMapper->RequestFinish();
	mpLoopCloser->RequestFinish();
	if(mpViewer)
	{
		mpViewer->RequestFinish();
		while(!mpViewer->isFinished())
			//usleep(5000);
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}

	// Wait until all thread have effectively stopped
	while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
	{
		//usleep(5000);
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}

	if(mpViewer)
		pangolin::BindToContext("MySLAM: Map Viewer");
}

void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
	cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

	vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
	sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

	// Transform all keyframes so that the first keyframe is at the origin.
	// After a loop closure the first keyframe might not be at the origin.
	//cv::Mat Two = vpKFs[0]->GetPoseInverse();

	ofstream f;
	f.open(filename.c_str());
	f << fixed;

	for(size_t i=0; i<vpKFs.size(); i++)
	{
		KeyFrame* pKF = vpKFs[i];

	   // pKF->SetPose(pKF->GetPose()*Two);

		if(pKF->isBad())
			continue;

		cv::Mat R = pKF->GetRotation().t();
		vector<float> q = Converter::toQuaternion(R);
		cv::Mat t = pKF->GetCameraCenter();
		f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
		  << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

	}

	f.close();
	cout << endl << "trajectory saved!" << endl;
}

int System::GetTrackingState()
{
	unique_lock<mutex> lock(mMutexState);
	return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
	unique_lock<mutex> lock(mMutexState);
	return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
	unique_lock<mutex> lock(mMutexState);
	return mTrackedKeyPointsUn;
}

} //namespace MySLAM
