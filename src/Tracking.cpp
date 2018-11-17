#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"

#include "Optimizer.h"
#include "PnPsolver.h"

#include <iostream>

#include <mutex>


using namespace std;

namespace MySLAM
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
	mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
	mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
	mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
	// Load camera parameters from settings file

	cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
	float fx = fSettings["Camera.fx"];
	float fy = fSettings["Camera.fy"];
	float cx = fSettings["Camera.cx"];
	float cy = fSettings["Camera.cy"];

	cv::Mat K = cv::Mat::eye(3,3,CV_32F);
	K.at<float>(0,0) = fx;
	K.at<float>(1,1) = fy;
	K.at<float>(0,2) = cx;
	K.at<float>(1,2) = cy;
	K.copyTo(mK);

	cv::Mat DistCoef(4,1,CV_32F);
	DistCoef.at<float>(0) = fSettings["Camera.k1"];
	DistCoef.at<float>(1) = fSettings["Camera.k2"];
	DistCoef.at<float>(2) = fSettings["Camera.p1"];
	DistCoef.at<float>(3) = fSettings["Camera.p2"];
	const float k3 = fSettings["Camera.k3"];
	if(k3!=0)
	{
		DistCoef.resize(5);
		DistCoef.at<float>(4) = k3;
	}
	DistCoef.copyTo(mDistCoef);

	mbf = fSettings["Camera.bf"];

	float fps = fSettings["Camera.fps"];
	if(fps==0)
		fps=30;

	// Max/Min Frames to insert keyframes and to check relocalisation
	mMinFrames = 0;
	mMaxFrames = fps;

	int nRGB = fSettings["Camera.RGB"];
	mbRGB = nRGB;

	// Load ORB parameters
	/* 每一帧提取的特征点数 1000 */
	int nFeatures = fSettings["ORBextractor.nFeatures"];
	/* 图像建立金字塔时的变化尺度 1.2 */
	float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
	/* 尺度金字塔的层数 8 */
	int nLevels = fSettings["ORBextractor.nLevels"];
	/* 提取FAST特征点的默认阈值 20 */
	int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
	/* 如果默认阈值提取不出足够FAST特征点，则使用最小阈值 8 */
	int fMinThFAST = fSettings["ORBextractor.minThFAST"];

	/* Tracking过程中用mpORBextractor提取特征点 */
	mpORBextractor = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
	/* 单目初始化时用mpIniORBextractor提取特征点 */
	mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
	mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
	mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
	mpViewer=pViewer;
}

/*
	输入左目RGB或RGBA图像
	1.将图像转为mImGray并初始化mCurrentFrame
	2.进行tracking过程
	输出世界坐标系到该帧相机坐标系的变换矩阵Tcw
*/
cv::Mat Tracking::GrabImage(const cv::Mat &im, const double &timestamp)
{
	/* 1.将图像转为mImGray并初始化mCurrentFrame */
	mImGray = im;
	if(mImGray.channels()==3)
	{
		if(mbRGB)
			cvtColor(mImGray,mImGray,CV_RGB2GRAY);
		else
			cvtColor(mImGray,mImGray,CV_BGR2GRAY);
	}
	else if(mImGray.channels()==4)
	{
		if(mbRGB)
			cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
		else
			cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
	}

	/* 2.构造Frame */
	if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
		mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
	else
		mCurrentFrame = Frame(mImGray,timestamp,mpORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

	/* 跟踪 */
	Track();

	return mCurrentFrame.mTcw.clone();
}

/* Tracking线程分为两个部分：相机位姿估计、跟踪局部地图 */
void Tracking::Track()
{
	/* 如果图像复位过、或者第一次运行，则为NO_IMAGE_YET状态 */
	if(mState==NO_IMAGES_YET)
		mState = NOT_INITIALIZED;

	mLastProcessedState=mState;/* 储存Tracking线程的最新状态，用于FrameDrawer的绘制 */

	// Get Map Mutex -> Map cannot be changed
	unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

	/* 初始化 */
	if(mState==NOT_INITIALIZED)
	{
		Initialization();
		mpFrameDrawer->Update(this);
		if(mState!=OK)
			return;
	}
	else/* 跟踪 */
	{
		// System is initialized. Track Frame.
		bool bOK;/* 临时变量，表示每个函数是否执行成功 */

		// Initial camera pose estimation using motion model or relocalization (if tracking is lost)
		if(!mbOnlyTracking)/* 跟踪+定位模式 */
		{
			// Local Mapping is activated. This is the normal behaviour, unless
			// you explicitly activate the "only tracking" mode.

			if(mState==OK)/* 正常初始化成功 */
			{
				// Local Mapping might have changed some MapPoints tracked in last frame
				/* 检查并更新上一帧被替换的MapPoints，更新Fuse函数和SearchAndFuse函数替换的MapPoints */
				CheckReplacedInLastFrame();
				
				/* 跟踪上一帧或者参考帧或者重定位 */
				if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
				{
					/* 若判定速度为空，或距离上次重定位少于2帧则用参考帧模型跟踪 */
					bOK = TrackReferenceKeyFrame();
				}
				else
				{
					/* 根据匀速运动模型计算初始位姿 */
					bOK = TrackWithMotionModel();
					/* 如果运动模型跟踪失败，使用参考帧模型进行跟踪 */
					if(!bOK)
						bOK = TrackReferenceKeyFrame();
				}
			}
			else
			{
				/* 跟踪丢失，用重定位找回当前相机的位姿 */
				bOK = Relocalization();
			}
		}
		else/* 只进行跟踪，局部地图不工作 */
		{
			// Localization Mode: Local Mapping is deactivated

			if(mState==LOST)
				bOK = Relocalization();
			else
			{
				/* mbOnlyTracking为true（只跟踪）时用到mbVO变量 */
				if(!mbVO)
				{
					// In last frame we tracked enough MapPoints in the map
					/* mbVO为false表示此帧匹配了很多的MapPoints，跟踪正常 */
					if(!mVelocity.empty())
						bOK = TrackWithMotionModel();
					else
						bOK = TrackReferenceKeyFrame();
				}
				else
				{
					// In last frame we tracked mainly "visual odometry" points.
					// We compute two camera poses, one from motion model and one doing relocalization.
					// If relocalization is sucessfull we choose that solution, otherwise we retain the "visual odometry" solution.
					/* mbVO为true表明此帧匹配了很少的MapPoints，跟踪即将失败，既做跟踪又做定位 */
					bool bOKMM = false;/* 跟踪是否成功 */
					bool bOKReloc = false;/* 重定位是否成功 */
					vector<MapPoint*> vpMPsMM;
					vector<bool> vbOutMM;
					cv::Mat TcwMM;
					if(!mVelocity.empty())
					{
						bOKMM = TrackWithMotionModel();
						vpMPsMM = mCurrentFrame.mvpMapPoints;
						vbOutMM = mCurrentFrame.mvbOutlier;
						TcwMM = mCurrentFrame.mTcw.clone();
					}
					bOKReloc = Relocalization();

					if(bOKMM && !bOKReloc)/* 跟踪成功但重定位没有成功 */
					{
						mCurrentFrame.SetPose(TcwMM);
						mCurrentFrame.mvpMapPoints = vpMPsMM;
						mCurrentFrame.mvbOutlier = vbOutMM;

						if(mbVO)
						{
							for(int i =0; i<mCurrentFrame.N; i++)
								if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
									mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
						}
					}
					else if(bOKReloc)/* 重定位成功则整个跟踪成功 */
						mbVO = false;

					bOK = bOKReloc || bOKMM;
				}
			}
		}

		/* 将最新的关键帧作为参考关键帧 */
		mCurrentFrame.mpReferenceKF = mpReferenceKF;

		// If we have an initial estimation of the camera pose and matching. Track the local map.
		if(!mbOnlyTracking)
		{
			if(bOK)
				bOK = TrackLocalMap();
		}
		else
		{
			// mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
			// a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
			// the camera we will use the local map again.
			if(bOK && !mbVO)
				bOK = TrackLocalMap();
		}

		if(bOK)
			mState = OK;
		else
			mState=LOST;

		// Update drawer
		mpFrameDrawer->Update(this);

		// If tracking were good, check if we insert a keyframe
		/* 跟踪成功，判断是否需要添加新的关键帧 */
		if(bOK)
		{
			// Update motion model
			if(!mLastFrame.mTcw.empty())
			{
				/* 更新匀速运动模型 */
				cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
				mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
				mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
				mVelocity = mCurrentFrame.mTcw*LastTwc;
			}
			else
				mVelocity = cv::Mat();

			mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

			// Clean VO matches
			/* 清除UpdateLastFrame中为当前帧临时添加的MapPoints */
			for(int i=0; i<mCurrentFrame.N; i++)
			{
				MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
				if(pMP)
					if(pMP->Observations()<1)
					{
						mCurrentFrame.mvbOutlier[i] = false;
						mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
					}
			}

			// Delete temporal MapPoints
			// Check if we need to insert a new keyframe
			if(NeedNewKeyFrame())
				CreateNewKeyFrame();

			// We allow points with high innovation (considererd outliers by the Huber Function)
			// pass to the new keyframe, so that bundle adjustment will finally decide
			// if they are outliers or not. We don't want next frame to estimate its position
			// with those points so we discard them in the frame.
			/* 删除在bundle adjustment中检测为outlier的3D map点 */
			for(int i=0; i<mCurrentFrame.N;i++)
			{
				if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
					mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
			}
		}

		// Reset if the camera get lost soon after initialization
		/* 跟踪失败且重定位也失败 */
		if(mState==LOST)
		{
			if(mpMap->KeyFramesInMap()<=5)
			{
				cout << "Tracking lost , reseting..." << endl;
				mpSystem->Reset();
				return;
			}
		}

		if(!mCurrentFrame.mpReferenceKF)
			mCurrentFrame.mpReferenceKF = mpReferenceKF;

		/* 保存上一帧 */
		mLastFrame = Frame(mCurrentFrame);
	}

	// Store frame pose information to retrieve the complete camera trajectory afterwards.
	if(!mCurrentFrame.mTcw.empty())
	{
		cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
		mlRelativeFramePoses.push_back(Tcr);
		mlpReferences.push_back(mpReferenceKF);
		mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
		mlbLost.push_back(mState==LOST);
	}
	else
	{
		// This can happen if tracking is lost
		mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
		mlpReferences.push_back(mlpReferences.back());
		mlFrameTimes.push_back(mlFrameTimes.back());
		mlbLost.push_back(mState==LOST);
	}

}

void Tracking::Initialization()
{

	if(!mpInitializer)
	{
		// Set Reference Frame
		if(mCurrentFrame.mvKeys.size()>100)
		{
			mInitialFrame = Frame(mCurrentFrame);
			mLastFrame = Frame(mCurrentFrame);
			mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
			for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
				mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

			if(mpInitializer)
				delete mpInitializer;

			mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

			fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

			return;
		}
	}
	else
	{
		// Try to initialize
		if((int)mCurrentFrame.mvKeys.size()<=100)
		{
			delete mpInitializer;
			mpInitializer = static_cast<Initializer*>(NULL);
			fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
			return;
		}

		// Find correspondences
		ORBmatcher matcher(0.9,true);
		int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

		// Check if there are enough correspondences
		if(nmatches<100)
		{
			delete mpInitializer;
			mpInitializer = static_cast<Initializer*>(NULL);
			return;
		}

		cv::Mat Rcw; // Current Camera Rotation
		cv::Mat tcw; // Current Camera Translation
		vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

		/* 通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints */
		if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
		{
			/* 删除无法进行三角测量的匹配点 */
			for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
			{
				if(mvIniMatches[i]>=0 && !vbTriangulated[i])
				{
					mvIniMatches[i]=-1;
					nmatches--;
				}
			}

			// Set Frame Poses
			/* 初始化的第一帧作为世界坐标系 */
			mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
			/* 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到该帧的变换矩阵 */
			cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
			Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
			tcw.copyTo(Tcw.rowRange(0,3).col(3));
			mCurrentFrame.SetPose(Tcw);

			/* 将三角化得到的3D点包装成MapPoints存入KeyFrame和Map中 */
			CreateInitialMap();
		}
	}
}

/* 三角测量生成MapPoints */
void Tracking::CreateInitialMap()
{
	// Create KeyFrames
	KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
	KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);


	pKFini->ComputeBoW();
	pKFcur->ComputeBoW();

	// Insert KFs in the map
	mpMap->AddKeyFrame(pKFini);
	mpMap->AddKeyFrame(pKFcur);

	// Create MapPoints and asscoiate to keyframes
	for(size_t i=0; i<mvIniMatches.size();i++)
	{
		if(mvIniMatches[i]<0)
			continue;

		//Create MapPoint.
		cv::Mat worldPos(mvIniP3D[i]);
		/* 用3D点构造MapPoint */
		MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

		/* 表示该KeyFrame的哪个特征点可以观测到哪个3D点 */
		pKFini->AddMapPoint(pMP,i);
		pKFcur->AddMapPoint(pMP,mvIniMatches[i]);
		/* 表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到 */
		pMP->AddObservation(pKFini,i);
		pMP->AddObservation(pKFcur,mvIniMatches[i]);

		/* 从众多观测到该MapPoint的特征点中挑选区分读最高的描述子 */
		pMP->ComputeDistinctiveDescriptors();
		/* 更新该MapPoint平均观测方向以及观测距离的范围 */
		pMP->UpdateNormalAndDepth();

		//Fill Current Frame structure
		mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
		mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

		//Add to Map
		/* 在地图中添加该MapPoint */
		mpMap->AddMapPoint(pMP);
	}

	// Update Connections
	/* 更新关键帧间的连接关系,在3D点和关键帧之间建立带权边，权重是该关键帧与当前帧公共3D点的个数 */
	pKFini->UpdateConnections();
	pKFcur->UpdateConnections();

	// Bundle Adjustment
	cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;
	/* BA优化 */
	Optimizer::GlobalBundleAdjustemnt(mpMap,20);

	// Set median depth to 1
	/* 将MapPoints的中值深度归一化到1，并归一化两帧之间变换 */
	float medianDepth = pKFini->ComputeSceneMedianDepth(2);/* 评估关键帧场景深度，q=2表示中值 */
	float invMedianDepth = 1.0f/medianDepth;

	if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
	{
		cout << "Wrong initialization, reseting..." << endl;
		Reset();
		return;
	}

	// Scale initial baseline
	cv::Mat Tc2w = pKFcur->GetPose();
	/* 将z归一化到1  */
	Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
	pKFcur->SetPose(Tc2w);

	// Scale points
	/* 把3D点的尺度也归一化到1 */
	vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
	for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
	{
		if(vpAllMapPoints[iMP])
		{
			MapPoint* pMP = vpAllMapPoints[iMP];
			pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
		}
	}

	mpLocalMapper->InsertKeyFrame(pKFini);
	mpLocalMapper->InsertKeyFrame(pKFcur);

	mCurrentFrame.SetPose(pKFcur->GetPose());
	mnLastKeyFrameId=mCurrentFrame.mnId;
	mpLastKeyFrame = pKFcur;

	mvpLocalKeyFrames.push_back(pKFcur);
	mvpLocalKeyFrames.push_back(pKFini);
	mvpLocalMapPoints=mpMap->GetAllMapPoints();
	mpReferenceKF = pKFcur;
	mCurrentFrame.mpReferenceKF = pKFcur;

	mLastFrame = Frame(mCurrentFrame);

	mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

	mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

	mpMap->mvpKeyFrameOrigins.push_back(pKFini);

	mState=OK;/* 初始化完成 */
}

/* LocalMapping线程可能会将关键帧中某些MapPoints进行替换，
	由于tracking中需要用到mLastFrame，这里检查并更新上一帧中被替换的MapPoints */
void Tracking::CheckReplacedInLastFrame()
{
	for(int i =0; i<mLastFrame.N; i++)
	{
		MapPoint* pMP = mLastFrame.mvpMapPoints[i];

		if(pMP)
		{
			MapPoint* pRep = pMP->GetReplaced();
			if(pRep)
			{
				mLastFrame.mvpMapPoints[i] = pRep;
			}
		}
	}
}

bool Tracking::TrackReferenceKeyFrame()
{
	// Compute Bag of Words vector
	/* 计算当前帧的BoW向量 */
	mCurrentFrame.ComputeBoW();

	// We perform first an ORB matching with the reference keyframe
	// If enough matches are found we setup a PnP solver
	/* 通过特征点的BoW加快当前帧与参考帧之间的特征点匹配，特征点的匹配关系由MapPoints进行维护 */
	ORBmatcher matcher(0.7,true);
	vector<MapPoint*> vpMapPointMatches;

	int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

	if(nmatches<15)
		return false;

	/* 将上一帧的位姿态作为当前帧位姿的初始值 */
	mCurrentFrame.mvpMapPoints = vpMapPointMatches;
	mCurrentFrame.SetPose(mLastFrame.mTcw);/* 用上一次的Tcw设置初值，在PoseOptimization可以收敛快一些 */

	/* 通过优化3D-2D的重投影误差来获得位姿 */
	Optimizer::PoseOptimization(&mCurrentFrame);

	// Discard outliers
	/* 剔除优化后的野值MapPoints */
	int nmatchesMap = 0;
	for(int i =0; i<mCurrentFrame.N; i++)
	{
		if(mCurrentFrame.mvpMapPoints[i])
		{
			if(mCurrentFrame.mvbOutlier[i])
			{
				MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

				mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
				mCurrentFrame.mvbOutlier[i]=false;
				pMP->mbTrackInView = false;
				pMP->mnLastFrameSeen = mCurrentFrame.mnId;
				nmatches--;
			}
			else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
				nmatchesMap++;
		}
	}

	return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
	// Update pose according to reference keyframe
	KeyFrame* pRef = mLastFrame.mpReferenceKF;
	cv::Mat Tlr = mlRelativeFramePoses.back();
	mLastFrame.SetPose(Tlr*pRef->GetPose());
}

bool Tracking::TrackWithMotionModel()
{
	ORBmatcher matcher(0.9,true);

	// Update last frame pose according to its reference keyframe
	// Create "visual odometry" points if in Localization Mode
	UpdateLastFrame();

	mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);/* 匀速运动模型 */

	fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

	// Project points seen in previous frame
	int th=15;
	int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,true);

	// If few matches, uses a wider window search
	if(nmatches<20)
	{
		fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
		nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,true);
	}

	if(nmatches<20)
		return false;

	// Optimize frame pose with all matches
	Optimizer::PoseOptimization(&mCurrentFrame);

	// Discard outliers
	int nmatchesMap = 0;
	for(int i =0; i<mCurrentFrame.N; i++)
	{
		if(mCurrentFrame.mvpMapPoints[i])
		{
			if(mCurrentFrame.mvbOutlier[i])
			{
				MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

				mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
				mCurrentFrame.mvbOutlier[i]=false;
				pMP->mbTrackInView = false;
				pMP->mnLastFrameSeen = mCurrentFrame.mnId;
				nmatches--;
			}
			else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
				nmatchesMap++;
		}
	}	

	if(mbOnlyTracking)
	{
		mbVO = nmatchesMap<10;
		return nmatches>20;
	}

	return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
	// We have an estimation of the camera pose and some map points tracked in the frame.
	// We retrieve the local map and try to find matches to points in the local map.

	UpdateLocalMap();

	SearchLocalPoints();

	// Optimize Pose
	Optimizer::PoseOptimization(&mCurrentFrame);
	mnMatchesInliers = 0;

	// Update MapPoints Statistics
	for(int i=0; i<mCurrentFrame.N; i++)
	{
		if(mCurrentFrame.mvpMapPoints[i])
		{
			if(!mCurrentFrame.mvbOutlier[i])
			{
				mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
				if(!mbOnlyTracking)
				{
					if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
						mnMatchesInliers++;
				}
				else
					mnMatchesInliers++;
			}
		}
	}

	// Decide if the tracking was succesful
	// More restrictive if there was a relocalization recently
	if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
		return false;

	if(mnMatchesInliers<30)
		return false;
	else
		return true;
}


bool Tracking::NeedNewKeyFrame()
{
	if(mbOnlyTracking)
		return false;

	// If Local Mapping is freezed by a Loop Closure do not insert keyframes
	if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
		return false;

	const int nKFs = mpMap->KeyFramesInMap();

	// Do not insert keyframes if not enough frames have passed from last relocalisation
	if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
		return false;

	// Tracked MapPoints in the reference keyframe
	int nMinObs = 3;
	if(nKFs<=2)
		nMinObs=2;
	int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

	// Local Mapping accept keyframes?
	bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

	// Check how many "close" points are being tracked and how many could be potentially created.
	int nNonTrackedClose = 0;
	int nTrackedClose= 0;

	bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

	// Thresholds
	float thRefRatio = 0.9f;
	// Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
	const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
	// Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
	const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
	// Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
	const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

	if((c1a||c1b)&&c2)
	{
		// If the mapping accepts keyframes, insert keyframe.
		// Otherwise send a signal to interrupt BA
		if(bLocalMappingIdle)
		{
			return true;
		}
		else
		{
			mpLocalMapper->InterruptBA();
			return false;
		}
	}
	else
		return false;
}

void Tracking::CreateNewKeyFrame()
{
	if(!mpLocalMapper->SetNotStop(true))
		return;
	KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
	mpReferenceKF = pKF;
	mCurrentFrame.mpReferenceKF = pKF;
	mpLocalMapper->InsertKeyFrame(pKF);
	mpLocalMapper->SetNotStop(false);
	mnLastKeyFrameId = mCurrentFrame.mnId;
	mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
	// Do not search map points already matched
	for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
	{
		MapPoint* pMP = *vit;
		if(pMP)
		{
			if(pMP->isBad())
			{
				*vit = static_cast<MapPoint*>(NULL);
			}
			else
			{
				pMP->IncreaseVisible();
				pMP->mnLastFrameSeen = mCurrentFrame.mnId;
				pMP->mbTrackInView = false;
			}
		}
	}

	int nToMatch=0;

	// Project points in frame and check its visibility
	for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
	{
		MapPoint* pMP = *vit;
		if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
			continue;
		if(pMP->isBad())
			continue;
		// Project (this fills MapPoint variables for matching)
		if(mCurrentFrame.isInFrustum(pMP,0.5))
		{
			pMP->IncreaseVisible();
			nToMatch++;
		}
	}

	if(nToMatch>0)
	{
		ORBmatcher matcher(0.8);
		int th = 1;
		// If the camera has been relocalised recently, perform a coarser search
		if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
			th=5;
		matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
	}
}

void Tracking::UpdateLocalMap()
{
	// This is for visualization
	mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

	// Update
	UpdateLocalKeyFrames();
	UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
	mvpLocalMapPoints.clear();

	for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
	{
		KeyFrame* pKF = *itKF;
		const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

		for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
		{
			MapPoint* pMP = *itMP;
			if(!pMP)
				continue;
			if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
				continue;
			if(!pMP->isBad())
			{
				mvpLocalMapPoints.push_back(pMP);
				pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
			}
		}
	}
}


void Tracking::UpdateLocalKeyFrames()
{
	// Each map point vote for the keyframes in which it has been observed
	map<KeyFrame*,int> keyframeCounter;
	for(int i=0; i<mCurrentFrame.N; i++)
	{
		if(mCurrentFrame.mvpMapPoints[i])
		{
			MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
			if(!pMP->isBad())
			{
				const map<KeyFrame*,size_t> observations = pMP->GetObservations();
				for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
					keyframeCounter[it->first]++;
			}
			else
			{
				mCurrentFrame.mvpMapPoints[i]=NULL;
			}
		}
	}

	if(keyframeCounter.empty())
		return;

	int max=0;
	KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

	mvpLocalKeyFrames.clear();
	mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

	// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
	for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
	{
		KeyFrame* pKF = it->first;

		if(pKF->isBad())
			continue;

		if(it->second>max)
		{
			max=it->second;
			pKFmax=pKF;
		}

		mvpLocalKeyFrames.push_back(it->first);
		pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
	}


	// Include also some not-already-included keyframes that are neighbors to already-included keyframes
	for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
	{
		// Limit the number of keyframes
		if(mvpLocalKeyFrames.size()>80)
			break;

		KeyFrame* pKF = *itKF;

		const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

		for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
		{
			KeyFrame* pNeighKF = *itNeighKF;
			if(!pNeighKF->isBad())
			{
				if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
				{
					mvpLocalKeyFrames.push_back(pNeighKF);
					pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
					break;
				}
			}
		}

		const set<KeyFrame*> spChilds = pKF->GetChilds();
		for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
		{
			KeyFrame* pChildKF = *sit;
			if(!pChildKF->isBad())
			{
				if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
				{
					mvpLocalKeyFrames.push_back(pChildKF);
					pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
					break;
				}
			}
		}

		KeyFrame* pParent = pKF->GetParent();
		if(pParent)
		{
			if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
			{
				mvpLocalKeyFrames.push_back(pParent);
				pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
				break;
			}
		}

	}

	if(pKFmax)
	{
		mpReferenceKF = pKFmax;
		mCurrentFrame.mpReferenceKF = mpReferenceKF;
	}
}

bool Tracking::Relocalization()
{
	// Compute Bag of Words Vector
	mCurrentFrame.ComputeBoW();

	// Relocalization is performed when tracking is lost
	// Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
	vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

	if(vpCandidateKFs.empty())
		return false;

	const int nKFs = vpCandidateKFs.size();

	// We perform first an ORB matching with each candidate
	// If enough matches are found we setup a PnP solver
	ORBmatcher matcher(0.75,true);

	vector<PnPsolver*> vpPnPsolvers;
	vpPnPsolvers.resize(nKFs);

	vector<vector<MapPoint*> > vvpMapPointMatches;
	vvpMapPointMatches.resize(nKFs);

	vector<bool> vbDiscarded;
	vbDiscarded.resize(nKFs);

	int nCandidates=0;

	for(int i=0; i<nKFs; i++)
	{
		KeyFrame* pKF = vpCandidateKFs[i];
		if(pKF->isBad())
			vbDiscarded[i] = true;
		else
		{
			int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
			if(nmatches<15)
			{
				vbDiscarded[i] = true;
				continue;
			}
			else
			{
				PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
				pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
				vpPnPsolvers[i] = pSolver;
				nCandidates++;
			}
		}
	}

	// Alternatively perform some iterations of P4P RANSAC
	// Until we found a camera pose supported by enough inliers
	bool bMatch = false;
	ORBmatcher matcher2(0.9,true);

	while(nCandidates>0 && !bMatch)
	{
		for(int i=0; i<nKFs; i++)
		{
			if(vbDiscarded[i])
				continue;

			// Perform 5 Ransac Iterations
			vector<bool> vbInliers;
			int nInliers;
			bool bNoMore;

			PnPsolver* pSolver = vpPnPsolvers[i];
			cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

			// If Ransac reachs max. iterations discard keyframe
			if(bNoMore)
			{
				vbDiscarded[i]=true;
				nCandidates--;
			}

			// If a Camera Pose is computed, optimize
			if(!Tcw.empty())
			{
				Tcw.copyTo(mCurrentFrame.mTcw);

				set<MapPoint*> sFound;

				const int np = vbInliers.size();

				for(int j=0; j<np; j++)
				{
					if(vbInliers[j])
					{
						mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
						sFound.insert(vvpMapPointMatches[i][j]);
					}
					else
						mCurrentFrame.mvpMapPoints[j]=NULL;
				}

				int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

				if(nGood<10)
					continue;

				for(int io =0; io<mCurrentFrame.N; io++)
					if(mCurrentFrame.mvbOutlier[io])
						mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

				// If few inliers, search by projection in a coarse window and optimize again
				if(nGood<50)
				{
					int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

					if(nadditional+nGood>=50)
					{
						nGood = Optimizer::PoseOptimization(&mCurrentFrame);

						// If many inliers but still not enough, search by projection again in a narrower window
						// the camera has been already optimized with many points
						if(nGood>30 && nGood<50)
						{
							sFound.clear();
							for(int ip =0; ip<mCurrentFrame.N; ip++)
								if(mCurrentFrame.mvpMapPoints[ip])
									sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
							nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

							// Final optimization
							if(nGood+nadditional>=50)
							{
								nGood = Optimizer::PoseOptimization(&mCurrentFrame);

								for(int io =0; io<mCurrentFrame.N; io++)
									if(mCurrentFrame.mvbOutlier[io])
										mCurrentFrame.mvpMapPoints[io]=NULL;
							}
						}
					}
				}


				// If the pose is supported by enough inliers stop ransacs and continue
				if(nGood>=50)
				{
					bMatch = true;
					break;
				}
			}
		}
	}

	if(!bMatch)
	{
		return false;
	}
	else
	{
		mnLastRelocFrameId = mCurrentFrame.mnId;
		return true;
	}

}

void Tracking::Reset()
{

	cout << "System Reseting" << endl;
	if(mpViewer)
	{
		mpViewer->RequestStop();
		while(!mpViewer->isStopped())
			this_thread::sleep_for(chrono::microseconds(3000));
	}

	// Reset Local Mapping
	cout << "Reseting Local Mapper...";
	mpLocalMapper->RequestReset();
	cout << " done" << endl;

	// Reset Loop Closing
	cout << "Reseting Loop Closing...";
	mpLoopClosing->RequestReset();
	cout << " done" << endl;

	// Clear BoW Database
	cout << "Reseting Database...";
	mpKeyFrameDB->clear();
	cout << " done" << endl;

	// Clear Map (this erase MapPoints and KeyFrames)
	mpMap->clear();

	KeyFrame::nNextId = 0;
	Frame::nNextId = 0;
	mState = NO_IMAGES_YET;

	if(mpInitializer)
	{
		delete mpInitializer;
		mpInitializer = static_cast<Initializer*>(NULL);
	}

	mlRelativeFramePoses.clear();
	mlpReferences.clear();
	mlFrameTimes.clear();
	mlbLost.clear();

	if(mpViewer)
		mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
	cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
	float fx = fSettings["Camera.fx"];
	float fy = fSettings["Camera.fy"];
	float cx = fSettings["Camera.cx"];
	float cy = fSettings["Camera.cy"];

	cv::Mat K = cv::Mat::eye(3,3,CV_32F);
	K.at<float>(0,0) = fx;
	K.at<float>(1,1) = fy;
	K.at<float>(0,2) = cx;
	K.at<float>(1,2) = cy;
	K.copyTo(mK);

	cv::Mat DistCoef(4,1,CV_32F);
	DistCoef.at<float>(0) = fSettings["Camera.k1"];
	DistCoef.at<float>(1) = fSettings["Camera.k2"];
	DistCoef.at<float>(2) = fSettings["Camera.p1"];
	DistCoef.at<float>(3) = fSettings["Camera.p2"];
	const float k3 = fSettings["Camera.k3"];
	if(k3!=0)
	{
		DistCoef.resize(5);
		DistCoef.at<float>(4) = k3;
	}
	DistCoef.copyTo(mDistCoef);

	mbf = fSettings["Camera.bf"];

	Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
	mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
