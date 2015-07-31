#include <stdio.h>
#include <string.h>
#include <time.h>
#include "BinTemplateCorrelation.cuh"

#define MAX_FILE_NAME_LENGTH 1000
#define MAX_FILE_LINE_LENGTH 100000

Cylinder *q;
unsigned int qLength;
Cylinder *db;
unsigned int dbLength;

unsigned int *templateDbLengthsTest;
unsigned int templateDbCountTest;

unsigned long long index;

void parseDb(char* path)
{
	FILE *file = fopen(path, "r");
	if (!file)
	{
		printf("DB open error\n");
		exit(EXIT_FAILURE);
	}

	char* line = new char[MAX_FILE_LINE_LENGTH];

	fgets(line, MAX_FILE_LINE_LENGTH, file);

	templateDbCountTest = strtoul(line, NULL, 10);
	templateDbLengthsTest = (unsigned int *)malloc(templateDbCountTest * sizeof(unsigned int));

	fgets(line, MAX_FILE_LINE_LENGTH, file);

	dbLength = 0;
	char *curTemplateDbLength = strtok(line, " ");
	for (unsigned int i = 0; i < templateDbCountTest; i++)
	{
		templateDbLengthsTest[i] = strtoul(curTemplateDbLength, NULL, 10);
		dbLength += templateDbLengthsTest[i];
		curTemplateDbLength = strtok(NULL, " ");
	}

	fgets(line, MAX_FILE_LINE_LENGTH, file);

	db = (Cylinder *)malloc(dbLength * sizeof(Cylinder));
	
	for (unsigned int i = 0; i < dbLength; i++)
	{
		Cylinder *curCylinder = new Cylinder(CYLINDER_CELLS_COUNT);

		fgets(line, MAX_FILE_LINE_LENGTH, file);
		createCylinderValues(line, CYLINDER_CELLS_COUNT * sizeof(unsigned int)* 8, curCylinder->values);

		fgets(line, MAX_FILE_LINE_LENGTH, file);
		curCylinder->angle = strtof(line, NULL);

		fgets(line, MAX_FILE_LINE_LENGTH, file);
		curCylinder->norm = strtof(line, NULL);

		fgets(line, MAX_FILE_LINE_LENGTH, file);
		curCylinder->templateIndex = strtol(line, NULL, 10);

		db[i] = *curCylinder;

		fgets(line, MAX_FILE_LINE_LENGTH, file);

		delete(curCylinder);
	}

	fclose(file);
}

void parseQ(char* path)
{
	FILE *file = fopen(path, "r");
	if (!file)
	{
		printf("Query open error\n");
		exit(EXIT_FAILURE);
	}

	char* line = new char[MAX_FILE_LINE_LENGTH];

	fgets(line, MAX_FILE_LINE_LENGTH, file); // Always 1 for query (number of templates)

	fgets(line, MAX_FILE_LINE_LENGTH, file);
	qLength = strtoul(line, NULL, 10);

	fgets(line, MAX_FILE_LINE_LENGTH, file);

	q = (Cylinder *)malloc(qLength * sizeof(Cylinder));
	
	for (unsigned int i = 0; i < qLength; i++)
	{
		Cylinder *curCylinder = new Cylinder(CYLINDER_CELLS_COUNT);

		fgets(line, MAX_FILE_LINE_LENGTH, file);
		createCylinderValues(line, CYLINDER_CELLS_COUNT * sizeof(unsigned int) * 8, curCylinder->values);

		fgets(line, MAX_FILE_LINE_LENGTH, file);
		curCylinder->angle = strtof(line, NULL);

		fgets(line, MAX_FILE_LINE_LENGTH, file);
		curCylinder->norm = strtof(line, NULL);

		fgets(line, MAX_FILE_LINE_LENGTH, file); // templateIndex (skip)

		q[i] = *curCylinder;

		fgets(line, MAX_FILE_LINE_LENGTH, file);

		delete(curCylinder);
	}

	fclose(file);
}

int main()
{
	char pathDb[MAX_FILE_NAME_LENGTH] = "C:\\Users\\resaglow\\mcc_c_db.txt";
	char pathQ[MAX_FILE_NAME_LENGTH] = "C:\\Users\\resaglow\\mcc_c_query.txt";

	parseDb(pathDb);
	parseQ(pathQ);

	initMCC(
		db, dbLength,
		templateDbLengthsTest, templateDbCountTest);

	clock_t start = clock();

	float* similarities = processMCC(
		q, qLength,
		dbLength, templateDbCountTest);

	cudaDeviceSynchronize();
	clock_t end = clock();
	printf("Global processing time: %ld\n", end - start);
	
	printf("Similarities:\n");
	for (unsigned int i = 0; i < templateDbCountTest; i++)
	{
		printf("%f%s", similarities[i], (i != templateDbCountTest - 1 ? "; " : ""));
	}
	printf("\n");

	free(db);
	free(q);

	return 0;
}