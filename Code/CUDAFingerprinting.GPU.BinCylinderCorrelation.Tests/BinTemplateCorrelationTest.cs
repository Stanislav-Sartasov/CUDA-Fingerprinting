using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common.BinCylinderCorrelation.Tests;
using CUDAFingerprinting.Common.BinCylinderCorrelation;

namespace CUDAFingerprinting.GPU.BinCylinderCorrelation.Tests
{
    [TestClass]
    public class BinTemplateCorrelationTest
    {
        [TestMethod]
        public void TestBinTemplateCorrelationMassive()
        {
            string homeFolder = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            BinTemplateCorrelationTests.ParseDb(homeFolder + "\\mcc_cs_db.txt");
            BinTemplateCorrelationTests.ParseQuery(homeFolder + "\\mcc_cs_query.txt");

            CylinderDatabase cylinderDb = new CylinderDatabase(
                BinTemplateCorrelationTests.contiguousCylinders, 
                BinTemplateCorrelationTests.templateIndices);

            double[] similarityRates = BinTemplateCorrelation.GetTemplateCorrelationMultipleOptimized(
                BinTemplateCorrelationTests.query, 
                cylinderDb, 
                BinTemplateCorrelationTests.templateDbLengths); // cylinderDb version

            for (int i = 0; i < similarityRates.Length; i++)
            {
                Console.Write(similarityRates[i] + (i != similarityRates.Length - 1 ? ", " : ""));
            }
            Console.WriteLine();
        }
    }
}
