#ifndef LOCAL_SIMILARITIES_SORT
#define LOCAL_SIMILARITIES_SORT

// Notes concerning data import:
//
// similaritiesDatabase (float* / short*) - 2D-array of local similarities scores
// with  query tempalte cylinders laid out on horizontal axis (column-coordinate)
// and all database templates cylinders on vertical axis (row-coordinate)
//
// templatesNumber (int) - the total number of templates in the database
//
// templateSizes (shortInt*) - the numbers of cylinders of each template in database
//
// queryTemplateSize (short) - the number of cylinders of the query template
//
// globalScores (float*) - preliminarily allocated output array

extern "C" {
	__declspec(dllexport) void getGlobalScoresFloat(
		float* globalScores,
		float* similaritiesDatabase,
		int templatesNumber,
		short* templateSizes,
		short queryTemplateSize);

	__declspec(dllexport) void getGlobalScoresShort(
		float* globalScores,
		short* similaritiesDatabase,
		int templatesNumber,
		short* templateSizes,
		short queryTemplateSize);
}

void getGlobalScoresFloat(
	float* globalScores,
	float* similaritiesDatabase,
	int templatesNumber,
	short* templateSizes,
	short queryTemplateSize);

void getGlobalScoresShort(
	float* globalScores,
	short* similaritiesDatabase,
	int templatesNumber,
	short* templateSizes,
	short queryTemplateSize);

#endif