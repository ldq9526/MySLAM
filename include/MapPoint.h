#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"

#include <opencv2/core/core.hpp>
#include <mutex>

namespace MySLAM
{

class KeyFrame;
class Map;
class Frame;


class MapPoint
{
public:
	MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
	MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);

	void SetWorldPos(const cv::Mat &Pos);
	cv::Mat GetWorldPos();

	cv::Mat GetNormal();
	KeyFrame* GetReferenceKeyFrame();

	std::map<KeyFrame*,size_t> GetObservations();
	int Observations();

	void AddObservation(KeyFrame* pKF,size_t idx);
	void EraseObservation(KeyFrame* pKF);

	int GetIndexInKeyFrame(KeyFrame* pKF);
	bool IsInKeyFrame(KeyFrame* pKF);

	void SetBadFlag();
	bool isBad();

	void Replace(MapPoint* pMP);	
	MapPoint* GetReplaced();

	void IncreaseVisible(int n=1);
	void IncreaseFound(int n=1);
	float GetFoundRatio();
	inline int GetFound(){
		return mnFound;
	}

	void ComputeDistinctiveDescriptors();

	cv::Mat GetDescriptor();

	void UpdateNormalAndDepth();

	float GetMinDistanceInvariance();
	float GetMaxDistanceInvariance();
	int PredictScale(const float &currentDist, KeyFrame*pKF);
	int PredictScale(const float &currentDist, Frame* pF);

public:
	long unsigned int mnId;/* 每个地图点有一个唯一的id */
	static long unsigned int nNextId;
	long int mnFirstKFid;/* 创建该地图点的关键帧KeyFrame的id */
	long int mnFirstFrame;/* 创建该地图点的帧Frame的id */
	int nObs;

	// Variables used by the tracking
	float mTrackProjX;
	float mTrackProjY;
	float mTrackProjXR;
	bool mbTrackInView;/* TrackLocalMap-SearchByProjection中决定是否对该点进行投影 */
	int mnTrackScaleLevel;
	float mTrackViewCos;
	long unsigned int mnTrackReferenceForFrame;/* TrackLocalMap-UpdateLocalPoints中防止将MapPoints重复添加至mvpLocalMapPoints */
	long unsigned int mnLastFrameSeen;/* TrackLocalMap-SearchLocalPoints中决定是否进行isInFrustum判断 */

	// Variables used by local mapping
	long unsigned int mnBALocalForKF;
	long unsigned int mnFuseCandidateForKF;

	// Variables used by loop closing
	long unsigned int mnLoopPointForKF;
	long unsigned int mnCorrectedByKF;
	long unsigned int mnCorrectedReference;	
	cv::Mat mPosGBA;/* global bundle adjustment优化的位姿 */
	long unsigned int mnBAGlobalForKF;


	static std::mutex mGlobalMutex;

protected:	

	 // Position in absolute coordinates
	 cv::Mat mWorldPos;/* 地图点的世界坐标 */

	 // Keyframes observing the point and associated index in keyframe
	 std::map<KeyFrame*,size_t> mObservations;/* 观测到该地图点的keyframe与该地图点在keyframe中的索引 */

	 // Mean viewing direction
	 cv::Mat mNormalVector;/* 该地图点的平均观测方向 */

	 // Best descriptor to fast matching
	 cv::Mat mDescriptor;/* 通过ComputeDistinctiveDescriptors()得到的最优描述子 */

	 // Reference KeyFrame
	 KeyFrame* mpRefKF;

	 // Tracking counters
	 int mnVisible;
	 int mnFound;

	 // Bad flag (we do not currently erase MapPoint from memory)
	 bool mbBad;
	 MapPoint* mpReplaced;

	 // Scale invariance distances
	 float mfMinDistance;
	 float mfMaxDistance;

	 Map* mpMap;

	 std::mutex mMutexPos;
	 std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
