#ifndef ORBVOCABULARY_H
#define ORBVOCABULARY_H

#include "3rdparty/DBoW2/DBoW2/FORB.h"
#include "3rdparty/DBoW2/DBoW2/TemplatedVocabulary.h"

namespace MySLAM
{
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;
} //namespace ORB_SLAM

#endif // ORBVOCABULARY_H
