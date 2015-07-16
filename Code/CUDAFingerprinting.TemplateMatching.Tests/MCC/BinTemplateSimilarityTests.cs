using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.TemplateMatching.MCC.Tests
{
    [TestClass]
    public class BinTemplateSimilarityTests
    {
        [TestMethod]
        public void TestBinTemplateCorrelation()
        {
            Cylinder cylinderZeros = CylinderTestsHelper.cylinderZeros;
            Cylinder cylinderOnes = CylinderTestsHelper.cylinderOnes;
            Cylinder cylinderMixed = CylinderTestsHelper.cylinderMixed;

            Template query = new Template(new Cylinder[]
            {
                cylinderMixed
            });

            Template[] db = new Template[]
            {
                new Template(new Cylinder[]
                {
                    cylinderOnes
                }),

                new Template(new Cylinder[]
                {
                    cylinderOnes,
                    cylinderMixed
                }),

                new Template(new Cylinder[]
                {
                    cylinderMixed,
                    cylinderMixed,
                    cylinderMixed,
                    cylinderMixed
                })
            };

            Cylinder[] contiguousArr = new Cylinder[]
            {
                cylinderOnes, cylinderOnes, cylinderMixed, cylinderMixed, cylinderMixed, cylinderMixed, cylinderMixed
            };
            uint[] templateIndices = new uint[] { 0, 1, 1, 2, 2, 2, 2 };
            CylinderDatabase cylinderDb = new CylinderDatabase(contiguousArr, templateIndices);
            int[] dbTemplatesLengths = new int[] { 1, 2, 4 };


            double[] similarityRates = BinTemplateSimilarity.GetTemplateSimilarity(query, db); // templateDb version
            //double[] similarityRates = BinCylinderSimilarity.GetTemplateSimilarityOptimized(query, cylinderDb, dbTemplatesLengths); // cylinderDb version
            for (int i = 0; i < similarityRates.Length; i++)
            {
                Console.Write(similarityRates[i] + (i != similarityRates.Length - 1 ? "; " : ""));
            }
            Console.WriteLine();
        }
    }
}
