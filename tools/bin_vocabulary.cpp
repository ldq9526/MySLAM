#include "ORBVocabulary.h"
using namespace std;

bool load_as_text(MySLAM::ORBVocabulary* voc, const std::string infile)
{
	bool res = voc->loadFromTextFile(infile);
	return res;
}

void load_as_xml(MySLAM::ORBVocabulary* voc, const std::string infile)
{
	voc->load(infile);
}

void load_as_binary(MySLAM::ORBVocabulary* voc, const std::string infile)
{
	voc->loadFromBinaryFile(infile);
}

void save_as_xml(MySLAM::ORBVocabulary* voc, const std::string outfile)
{
	voc->save(outfile);
}

void save_as_text(MySLAM::ORBVocabulary* voc, const std::string outfile)
{
	voc->saveToTextFile(outfile);
}

void save_as_binary(MySLAM::ORBVocabulary* voc, const std::string outfile)
{
	voc->saveToBinaryFile(outfile);
}


int main(int argc, char **argv)
{
	cout << "BoW load/save benchmark" << endl;
	MySLAM::ORBVocabulary* voc = new MySLAM::ORBVocabulary();

	load_as_text(voc, argv[1]);
	save_as_binary(voc, argv[2]);

	return 0;
}

