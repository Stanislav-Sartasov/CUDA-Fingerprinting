#ifndef LOCAL_SIMILARITIES_SORT
#define LOCAL_SIMILARITIES_SORT

// Notes concerning data import:
// similaritiesDatabase (float* / short*) - 2D-array of local similarities scores
// templatesNumber (int) - the total number of templates in the database
// templateSizes (shortInt*) - the numbers of cylinders of each template in database
// queryTemplateSize (short) - the number of cylinders of the query template
// globalScores (float*) - preliminarily allocated output array

extern "C" {
	__declspec(dllexport) void getGlobalScoresFloat(
		//float* similaritiesDatabase[],
		float* similaritiesDatabase,
		int templatesNumber,
		short* templateSizes,
		short queryTemplateSize,
		float* globalScores);

	__declspec(dllexport) void getGlobalScoresShort(
		//short* similaritiesDatabase[],
		short* similaritiesDatabase,
		int templatesNumber,
		short* templateSizes,
		short queryTemplateSize,
		float* globalScores);
}

void getGlobalScoresFloat(
	//float* similaritiesDatabase[],
	float* similaritiesDatabase,
	int templatesNumber,
	short* templateSizes,
	short queryTemplateSize,
	float* globalScores);

void getGlobalScoresShort(
	//short* similaritiesDatabase[],
	short* similaritiesDatabase,
	int templatesNumber,
	short* templateSizes,
	short queryTemplateSize,
	float* globalScores);

#endif