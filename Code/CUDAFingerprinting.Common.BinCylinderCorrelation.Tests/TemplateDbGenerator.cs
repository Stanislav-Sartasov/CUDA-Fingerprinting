using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;

namespace CUDAFingerprinting.Common.BinCylinderCorrelation.Tests
{
    [TestClass]
    public class TemplateDbGenerator
    {
        // Storage
        public static Cylinder[] db;
        public static int[] templateIndices;
        public static int[] templateDbLengths;

        // Params
        public static int cylinderDbCount;
        public static int templateDbCount;
        public static int cylinderCellsCount;

        public static Random rnd = new Random();

        public static void GenerateTemplateDb(
            int givenCylinderDbCount, int givenTemplateDbCount, int givenCylinderCellsCount)
        {
            cylinderDbCount = givenCylinderDbCount;
            templateDbCount = givenTemplateDbCount;
            cylinderCellsCount = givenCylinderCellsCount;

            db = new Cylinder[cylinderDbCount];
            templateIndices = new int[cylinderDbCount];
            templateDbLengths = new int[templateDbCount];

            for (int i = 0; i < cylinderDbCount; i++)
            {
                Cylinder curCylinder = new Cylinder();

                // For further randomness (so that cylinders don't have very similar norms)
                double curThreshold = rnd.NextDouble();

                uint[] curCylinderValues = new uint[cylinderCellsCount];
                for (int j = 0; j < cylinderCellsCount; j++)
                {
                    double x = rnd.NextDouble();
                    curCylinderValues[j] = x < curThreshold ? (uint)0 : 1; // Cast necessary? o_o
                }
                curCylinder.Values = curCylinderValues;

                curCylinder.Angle = rnd.NextDouble() * 2 * Math.PI;
                curCylinder.Norm = CylinderHelper.CalculateCylinderNorm(curCylinder.Values);

                db[i] = curCylinder;
            }

            // Thresholds again for further randomness (not a uniform distribution between templates)
            double[] templateThresholds = new double[templateDbCount - 1];
            for (int i = 0; i < templateDbCount - 1; i++)
            {
                templateThresholds[i] = rnd.NextDouble();
            }

            for (int i = 0; i < cylinderDbCount; i++)
            {
                double x = rnd.NextDouble();

                int curTemplateIndex = 0;
                for (int j = 0; j < templateDbCount - 1; j++)
                {
                    if (x > templateThresholds[j])
                    {
                        curTemplateIndex++;
                    }
                }

                templateIndices[i] = curTemplateIndex;
                templateDbLengths[curTemplateIndex]++;
            }
        }

        public static void WriteDbToFile(string pathCS, string pathC)
        {
            using (var file1 = new StreamWriter(pathCS))
            using (var file2 = new StreamWriter(pathC))
            {
                file2.WriteLine(templateDbCount);

                for (int i = 0; i < templateDbCount; i++)
                {
                    file1.Write(templateDbLengths[i] + (i != templateDbCount - 1 ? " " : ""));
                    file2.Write(templateDbLengths[i] + (i != templateDbCount - 1 ? " " : ""));
                }
                file1.WriteLine();
                file2.WriteLine();

                // 2 versions of templateIndices places (only 1 should be used at a time!)
                // C#-compatible version
                for (int i = 0; i < cylinderDbCount; i++)
                {
                    file1.Write(templateIndices[i] + (i != cylinderDbCount - 1 ? " " : ""));
                }
                file1.WriteLine();
                // [end] C#-compatible version

                file1.WriteLine();
                file2.WriteLine();

                for (int i = 0; i < cylinderDbCount; i++)
                {
                    for (int j = 0; j < cylinderCellsCount; j++)
                    {
                        file1.Write(db[i].Values[j]);
                        file2.Write(db[i].Values[j]);
                    }
                    file1.WriteLine();
                    file1.WriteLine(db[i].Angle);
                    file1.WriteLine(db[i].Norm);
                    file1.WriteLine();

                    file2.WriteLine();
                    file2.WriteLine(db[i].Angle);
                    file2.WriteLine(db[i].Norm);
                    file2.WriteLine(templateIndices[i]); // C-compatible version
                    file2.WriteLine();
                }
            }
        }

        [TestMethod]
        public void TestTemplateDbGenerator()
        {
            Random rnd = new Random();

            string homeFolder = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            GenerateTemplateDb(10000, 300, 255);
            WriteDbToFile(homeFolder + "\\mcc_cs_db.txt", homeFolder + "\\mcc_c_db.txt");

            GenerateTemplateDb(32, 1, 255); // 1 cylinder for query (it necessary for algorithm to work)
            WriteDbToFile(homeFolder + "\\mcc_cs_query.txt", homeFolder + "\\mcc_c_query.txt");
        }
    }
}
