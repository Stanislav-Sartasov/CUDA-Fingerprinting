using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.IO;

namespace CUDAFingerprinting.TemplateMatching.MCC.Tests
{
    [TestClass]
    public class BinTemplateSimilarityTests
    {
        public static Cylinder cylinderZeros = CylinderTestsHelper.cylinderZeros;
        public static Cylinder cylinderOnes = CylinderTestsHelper.cylinderOnes;
        public static Cylinder cylinderMixed = CylinderTestsHelper.cylinderMixed;

        public static Cylinder[] cylinders = new Cylinder[]
        {
            CylinderTestsHelper.cylinderZeros,
            CylinderTestsHelper.cylinderOnes,
            CylinderTestsHelper.cylinderMixed
        };

        public static Template query;

        public static Template[] db; // DB for simple version

        public static Cylinder[] contiguousCylinders; // DB for optimized version
        public static uint[] templateIndices;
        public static CylinderDatabase cylinderDb;
        public static int[] templateDbLengths;

        [TestMethod]
        public void TestBinTemplateCorrelation()
        {
            query = new Template(new Cylinder[] 
            {
                cylinderMixed
            });

            db = new Template[]
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

            contiguousCylinders = new Cylinder[]
            {
                cylinderOnes, cylinderOnes, cylinderMixed, cylinderMixed, cylinderMixed, cylinderMixed, cylinderMixed
            };
            templateIndices = new uint[] { 0, 1, 1, 2, 2, 2, 2 };
            cylinderDb = new CylinderDatabase(contiguousCylinders, templateIndices);
            templateDbLengths = new int[] { 1, 2, 4 };

            double[] similarityRates = BinTemplateSimilarity.GetTemplateSimilarity(query, db); // templateDb version
            //double[] similarityRates = BinCylinderSimilarity.GetTemplateSimilarityOptimized(query, cylinderDb, dbTemplatesLengths); // cylinderDb version
            
            for (int i = 0; i < similarityRates.Length; i++)
            {
                Console.Write(similarityRates[i] + (i != similarityRates.Length - 1 ? "; " : ""));
            }
            Console.WriteLine();
        }

        public static void ParseDb(string path)
        {
            using (var file = new StreamReader(path))
            {
                string[] templateDbLengthsString = file.ReadLine().Split(new Char[] { ' ' });
                templateDbLengths = new int[templateDbLengthsString.Length];
                for (int i = 0; i < templateDbLengthsString.Length; i++)
                {

                    templateDbLengths[i] = Int32.Parse(templateDbLengthsString[i]);
                }

                string[] templateIndicesString = file.ReadLine().Split(new Char[] { ' ' });
                templateIndices = new uint[templateIndicesString.Length];
                for (int i = 0; i < templateIndicesString.Length; i++)
                {
                    templateIndices[i] = UInt32.Parse(templateIndicesString[i]);
                }

                file.ReadLine();

                contiguousCylinders = new Cylinder[templateIndices.Length];
                for (int i = 0; i < templateIndices.Length; i++)
                {
                    Cylinder curCylinder = new Cylinder();

                    string curCylinderString = file.ReadLine();
                    uint[] curCylinderUInt = new uint[curCylinderString.Length];
                    for (int j = 0; j < curCylinderUInt.Length; j++)
                    {
                        curCylinderUInt[j] = UInt32.Parse(curCylinderString[j].ToString());
                    }
                    curCylinder.Values = CylinderTestsHelper.ConvertArrayUintToBinary(curCylinderUInt);

                    curCylinder.Angle = Double.Parse(file.ReadLine());
                    curCylinder.Norm = Double.Parse(file.ReadLine());

                    contiguousCylinders[i] = curCylinder;

                    file.ReadLine();
                }
            }
        }

        public static void ParseQuery(string path)
        {
            using (var file = new StreamReader(path))
            {
                Cylinder[] queryCylinders = new Cylinder[Int32.Parse(file.ReadLine())];
                file.ReadLine();
                file.ReadLine();

                for (int i = 0; i < queryCylinders.Length; i++)
                {
                    Cylinder curCylinder = new Cylinder();

                    string curCylinderString = file.ReadLine();
                    uint[] curCylinderUInt = new uint[curCylinderString.Length];
                    for (int j = 0; j < curCylinderUInt.Length; j++)
                    {
                        curCylinderUInt[j] = UInt32.Parse(curCylinderString[j].ToString());
                    }
                    curCylinder.Values = CylinderTestsHelper.ConvertArrayUintToBinary(curCylinderUInt);

                    curCylinder.Angle = Double.Parse(file.ReadLine());
                    curCylinder.Norm = Double.Parse(file.ReadLine());

                    queryCylinders[i] = curCylinder;

                    file.ReadLine();
                }

                query = new Template(queryCylinders);
            }
        }
        
        [TestMethod]
        public void TestBinTemplateSimilarityMassive()
        {
            string homeFolder = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            ParseDb(homeFolder + "\\mcc_cs_db.txt");
            ParseQuery(homeFolder + "\\mcc_cs_query.txt");

            CylinderDatabase cylinderDb = new CylinderDatabase(contiguousCylinders, templateIndices);

            double[] similarityRates = BinTemplateSimilarity.GetTemplateSimilarityOptimized(query, cylinderDb, templateDbLengths); // cylinderDb version
            for (int i = 0; i < similarityRates.Length; i++)
            {
                Console.Write(similarityRates[i] + (i != similarityRates.Length - 1 ? ", " : ""));
            }
            Console.WriteLine();
        }
    }
}
