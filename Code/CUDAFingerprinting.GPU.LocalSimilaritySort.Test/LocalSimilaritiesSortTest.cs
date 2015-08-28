using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;
using System.IO;

namespace CUDAFingerprinting.GPU.LocalSimilaritySort.Test
{
    [TestClass]
    public class LocalSimilaritiesSortTest
    {
        [DllImport("CUDAFingerprinting.GPU.LocalSimilaritySort.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "getGlobalScoresFloat")]
        public static extern void computeGlobalScores(
            float[] globalScores,
            float[,] similaritiesDatabase,
            int templatesNumber,
            short[] templateSizes,
            short queryTemplateSize);

        [DllImport("CUDAFingerprinting.GPU.LocalSimilaritySort.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "getGlobalScoresShort")]
        public static extern void computeGlobalScores(
            float[] globalScores,
            short[,] similaritiesDatabase,
            int templatesNumber,
            short[] templateSizes,
            short queryTemplateSize);

        [TestMethod]
        public void TestLssWithFloat()
        {
            var database = new LocalScoresTable<float>();
            database.RandGenerate();
            //database.ReadFile();

            DateTime workingTime = DateTime.Now;
            float[] globalScores = new float[database.TemplatesNumber];
            for (int i = 0; i < 30; ++i)
            {
                if (i == 10) workingTime = DateTime.Now;
                computeGlobalScores(globalScores, database.data, database.TemplatesNumber, database.templateSizes, database.QueryTemplateSize);
            }
            TimeSpan difference = (DateTime.Now - workingTime);

            int averageTime = difference.Milliseconds / 20;

            bool plausibleResult = true;
            float scoreSum = 0;
            foreach(float score in globalScores)
            {
                if((score < 0.0) || (score > 1.0))
                    plausibleResult = false; 
                scoreSum += score;
            }
            if(scoreSum < globalScores.GetLength(0) * 0.0001)
                plausibleResult = false;

            Assert.IsTrue(plausibleResult);
        }

        [TestMethod]
        public void TestLssWithShort()
        {
            var database = new LocalScoresTable<short>();
            database.RandGenerate();
            //database.ReadFile();

            DateTime workingTime = DateTime.Now;
            float[] globalScores = new float[database.TemplatesNumber];
            for (int i = 0; i < 30; ++i)
            {
                if (i == 10) workingTime = DateTime.Now;
                computeGlobalScores(globalScores, database.data, database.TemplatesNumber, database.templateSizes, database.QueryTemplateSize);
            }
            TimeSpan difference = (DateTime.Now - workingTime);

            int averageTime = difference.Milliseconds / 20;

            bool plausibleResult = true;
            float scoreSum = 0;
            foreach (float score in globalScores)
            {
                if ((score < 0.0) || (score > 1.0))
                    plausibleResult = false;
                scoreSum += score;
            }
            if (scoreSum < globalScores.GetLength(0) * 0.0001)
                plausibleResult = false;

            Assert.IsTrue(plausibleResult);
        }

        //static public void Main()
        //{
        //    var database = new LocalScoresTable<float>();

        //    database.RandGenerate();

        //    database.SaveDBToFile(@"C:\GitHub\CUDA-Fingerprinting\Code\CUDAFingerprinting.GPU.LocalSimilaritySort\database.txt");
        //    database.SaveTemplateSizesToFile(@"C:\GitHub\CUDA-Fingerprinting\Code\CUDAFingerprinting.GPU.LocalSimilaritySort\templateSizes.txt");
        //}
    }
}
