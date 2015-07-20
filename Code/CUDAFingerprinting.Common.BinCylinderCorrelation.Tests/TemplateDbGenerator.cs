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

        public static void GenerateTemplateDb(
            int givenCylinderDbCount, int givenTemplateDbCount, int givenCylinderCellsCount)
        {
            cylinderDbCount = givenCylinderDbCount;
            templateDbCount = givenTemplateDbCount;
            cylinderCellsCount = givenCylinderCellsCount;

            db = new Cylinder[cylinderDbCount];
            templateIndices = new int[cylinderDbCount];
            templateDbLengths = new int[templateDbCount];

            Random rnd = new Random();

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

        public static void WriteDbToFile(string path)
        {
            using (var file = new StreamWriter(path))
            {
                for (int i = 0; i < templateDbCount; i++)
                {
                    file.Write(templateDbLengths[i] + (i != templateDbCount - 1 ? " " : ""));
                }
                file.WriteLine();

                // 2 versions of templateIndices places (only 1 should be used at a time!)
                // C#-compatible version
                for (int i = 0; i < cylinderDbCount; i++)
                {
                    file.Write(templateIndices[i] + (i != cylinderDbCount - 1 ? " " : ""));
                }
                file.WriteLine();
                // [end] C#-compatible version

                file.WriteLine();

                for (int i = 0; i < cylinderDbCount; i++)
                {
                    for (int j = 0; j < cylinderCellsCount; j++)
                    {
                        file.Write(db[i].Values[j]);
                    }
                    file.WriteLine();
                    file.WriteLine(db[i].Angle);
                    file.WriteLine(db[i].Norm);
                    //file.WriteLine(templateIndices[i]); // C-compatible version
                    file.WriteLine();
                }
            }
        }

        [TestMethod]
        public void TestTemplateDbGenerator()
        {
            string homeFolder = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            GenerateTemplateDb(2000, 25, 255);
            WriteDbToFile(homeFolder + "\\mcc_db.txt");

            GenerateTemplateDb(100000, 1, 255); // 1 cylinder for query
            WriteDbToFile(homeFolder + "\\mcc_query.txt");
        }
    }
}
