#include "MapPoint.h"
#include "ORBmatcher.h"

#include <mutex>

namespace MySLAM
{

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
	mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
	mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
	mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
	mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
	Pos.copyTo(mWorldPos);/* 世界坐标 */
	mNormalVector = cv::Mat::zeros(3,1,CV_32F);

	// MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
	unique_lock<mutex> lock(mpMap->mMutexPointCreation);
	mnId=nNextId++;
}

MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
	mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
	mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
	mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
	mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
	Pos.copyTo(mWorldPos);
	cv::Mat Ow = pFrame->GetCameraCenter();/* 世界坐标系下的相机中心坐标 */
	mNormalVector = mWorldPos - Ow;/* 世界坐标系下相机中心到地图点的向量 */
	mNormalVector = mNormalVector/cv::norm(mNormalVector);/* 单位化观察方向向量 */

	cv::Mat PC = Pos - Ow;
	const float dist = cv::norm(PC);
	const int level = pFrame->mvKeysUn[idxF].octave;
	const float levelScaleFactor =  pFrame->mvScaleFactors[level];
	const int nLevels = pFrame->mnScaleLevels;

	mfMaxDistance = dist*levelScaleFactor;
	mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

	pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

	// MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
	unique_lock<mutex> lock(mpMap->mMutexPointCreation);
	mnId=nNextId++;
}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
	unique_lock<mutex> lock2(mGlobalMutex);
	unique_lock<mutex> lock(mMutexPos);
	Pos.copyTo(mWorldPos);
}

cv::Mat MapPoint::GetWorldPos()
{
	unique_lock<mutex> lock(mMutexPos);
	return mWorldPos.clone();
}

cv::Mat MapPoint::GetNormal()
{
	unique_lock<mutex> lock(mMutexPos);
	return mNormalVector.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame()
{
	unique_lock<mutex> lock(mMutexFeatures);
	return mpRefKF;
}

/* 记录能观测到该地图点的关键帧以及相应的特征点 */
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
	unique_lock<mutex> lock(mMutexFeatures);
	if(mObservations.count(pKF))
		return;
	/* 能观测到该地图点的关键帧与特征点索引 */
	mObservations[pKF]=idx;
	++nObs;
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
	bool bBad=false;
	{
		unique_lock<mutex> lock(mMutexFeatures);
		if(mObservations.count(pKF))
		{
			--nObs;
			mObservations.erase(pKF);
			/* 如果该关键帧为参考帧，需要制定新的参考帧 */
			if(mpRefKF==pKF)
				mpRefKF=mObservations.begin()->first;

			// If only 2 observations or less, discard point
			if(nObs<=2)
				bBad=true;
		}
	}

	if(bBad)
		SetBadFlag();
}

map<KeyFrame*, size_t> MapPoint::GetObservations()
{
	unique_lock<mutex> lock(mMutexFeatures);
	return mObservations;
}

int MapPoint::Observations()
{
	unique_lock<mutex> lock(mMutexFeatures);
	return nObs;
}

/* 通知能观测到该地图点的Frame删除该点 */
void MapPoint::SetBadFlag()
{
	map<KeyFrame*,size_t> obs;
	{
		unique_lock<mutex> lock1(mMutexFeatures);
		unique_lock<mutex> lock2(mMutexPos);
		mbBad=true;
		obs = mObservations;
		mObservations.clear();
	}
	for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
	{
		KeyFrame* pKF = mit->first;
		pKF->EraseMapPointMatch(mit->second);
	}

	mpMap->EraseMapPoint(this);
}

MapPoint* MapPoint::GetReplaced()
{
	unique_lock<mutex> lock1(mMutexFeatures);
	unique_lock<mutex> lock2(mMutexPos);
	return mpReplaced;
}

/* 形成闭环时，更新KeyFrame与MapPoint之间的关系 */
void MapPoint::Replace(MapPoint* pMP)
{
	if(pMP->mnId==this->mnId)
		return;

	int nvisible, nfound;
	map<KeyFrame*,size_t> obs;
	{
		unique_lock<mutex> lock1(mMutexFeatures);
		unique_lock<mutex> lock2(mMutexPos);
		obs=mObservations;
		mObservations.clear();
		mbBad=true;
		nvisible = mnVisible;
		nfound = mnFound;
		mpReplaced = pMP;
	}

	/* 所有能观测到该MapPoint的keyframe都要替换 */
	for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
	{
		// Replace measurement in keyframe
		KeyFrame* pKF = mit->first;

		if(!pMP->IsInKeyFrame(pKF))
		{
			pKF->ReplaceMapPointMatch(mit->second, pMP);
			pMP->AddObservation(pKF,mit->second);
		}
		else
		{
			pKF->EraseMapPointMatch(mit->second);
		}
	}
	pMP->IncreaseFound(nfound);
	pMP->IncreaseVisible(nvisible);
	pMP->ComputeDistinctiveDescriptors();

	mpMap->EraseMapPoint(this);
}

/* 该地图点是否经过MapPointCulling检测 */
bool MapPoint::isBad()
{
	unique_lock<mutex> lock(mMutexFeatures);
	unique_lock<mutex> lock2(mMutexPos);
	return mbBad;
}

void MapPoint::IncreaseVisible(int n)
{
	unique_lock<mutex> lock(mMutexFeatures);
	mnVisible+=n;
}

void MapPoint::IncreaseFound(int n)
{
	unique_lock<mutex> lock(mMutexFeatures);
	mnFound+=n;
}

float MapPoint::GetFoundRatio()
{
	unique_lock<mutex> lock(mMutexFeatures);
	return static_cast<float>(mnFound)/mnVisible;
}

void MapPoint::ComputeDistinctiveDescriptors()
{
	// Retrieve all observed descriptors
	vector<cv::Mat> vDescriptors;

	map<KeyFrame*,size_t> observations;

	{
		unique_lock<mutex> lock1(mMutexFeatures);
		if(mbBad)
			return;
		observations=mObservations;
	}

	if(observations.empty())
		return;

	vDescriptors.reserve(observations.size());

	for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
	{
		KeyFrame* pKF = mit->first;

		if(!pKF->isBad())
			vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
	}

	if(vDescriptors.empty())
		return;

	// Compute distances between them
	const size_t N = vDescriptors.size();

	vector<vector<float> > Distances(N, vector<float>(N, 0));
	for(size_t i=0;i<N;i++)
	{
		Distances[i][i]=0;
		for(size_t j=i+1;j<N;j++)
		{
			int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
			Distances[i][j]=distij;
			Distances[j][i]=distij;
		}
	}

	// Take the descriptor with least median distance to the rest
	int BestMedian = INT_MAX;
	int BestIdx = 0;
	for(size_t i=0;i<N;i++)
	{
		vector<int> vDists(Distances[i].begin(), Distances[i].end());
		sort(vDists.begin(),vDists.end());
		int median = vDists[0.5*(N-1)];

		if(median<BestMedian)
		{
			BestMedian = median;
			BestIdx = i;
		}
	}

	{
		unique_lock<mutex> lock(mMutexFeatures);
		mDescriptor = vDescriptors[BestIdx].clone();
	}
}

cv::Mat MapPoint::GetDescriptor()
{
	unique_lock<mutex> lock(mMutexFeatures);
	return mDescriptor.clone();
}

int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
	unique_lock<mutex> lock(mMutexFeatures);
	if(mObservations.count(pKF))
		return mObservations[pKF];
	else
		return -1;
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
	unique_lock<mutex> lock(mMutexFeatures);
	return (mObservations.count(pKF));
}

void MapPoint::UpdateNormalAndDepth()
{
	map<KeyFrame*,size_t> observations;
	KeyFrame* pRefKF;
	cv::Mat Pos;
	{
		unique_lock<mutex> lock1(mMutexFeatures);
		unique_lock<mutex> lock2(mMutexPos);
		if(mbBad)
			return;
		observations=mObservations;
		pRefKF=mpRefKF;
		Pos = mWorldPos.clone();
	}

	if(observations.empty())
		return;

	cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
	int n=0;
	for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
	{
		KeyFrame* pKF = mit->first;
		cv::Mat Owi = pKF->GetCameraCenter();
		cv::Mat normali = mWorldPos - Owi;
		normal = normal + normali/cv::norm(normali);
		n++;
	}

	cv::Mat PC = Pos - pRefKF->GetCameraCenter();
	const float dist = cv::norm(PC);
	const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
	const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
	const int nLevels = pRefKF->mnScaleLevels;

	{
		unique_lock<mutex> lock3(mMutexPos);
		mfMaxDistance = dist*levelScaleFactor;
		mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
		mNormalVector = normal/n;
	}
}

float MapPoint::GetMinDistanceInvariance()
{
	unique_lock<mutex> lock(mMutexPos);
	return 0.8f*mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
	unique_lock<mutex> lock(mMutexPos);
	return 1.2f*mfMaxDistance;
}

int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
	float ratio;
	{
		unique_lock<mutex> lock(mMutexPos);
		ratio = mfMaxDistance/currentDist;
	}

	int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
	if(nScale<0)
		nScale = 0;
	else if(nScale>=pKF->mnScaleLevels)
		nScale = pKF->mnScaleLevels-1;

	return nScale;
}

int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
	float ratio;
	{
		unique_lock<mutex> lock(mMutexPos);
		ratio = mfMaxDistance/currentDist;
	}

	int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
	if(nScale<0)
		nScale = 0;
	else if(nScale>=pF->mnScaleLevels)
		nScale = pF->mnScaleLevels-1;

	return nScale;
}



} //namespace ORB_SLAM
