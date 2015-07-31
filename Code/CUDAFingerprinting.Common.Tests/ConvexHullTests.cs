using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.IO;

namespace CUDAFingerprinting.Common.ConvexHull.Tests
{
    [TestClass]
    public class ConvexHullTests
    {
        public static int bmpX = 1100, bmpY = 1100;
        public static double omega = 40;

        public static List<PointF> globalHull = new List<PointF>(
            new PointF[]
            {
                new PointF(0, 100),
                new PointF(200, 0),
                new PointF(800, 300),
                new PointF(600, 600),
                new PointF(100, 900),
            });

        public static List<PointF> globalHullMassive;

        // input file should be in format:
        // <num of points>
        // x, y
        // x, y
        // ...
        // Could be generated at random.org/integer-sets/
        public static void ParsePoints(string path)
        {
            using (var file = new StreamReader(path))
            {
                int pointCount = Int32.Parse(file.ReadLine());

                globalHullMassive = new List<PointF>();

                for (int i = 0; i < pointCount; i++)
                {
                    string[] pointStr = file.ReadLine().Split(new Char[] { ',', ' ' });
                    // [2] because string gets split like "x", " ", "y", don't know how to "fix" it
                    float x = Single.Parse(pointStr[0]), y = Single.Parse(pointStr[2]);
                    PointF curPoint = new PointF(x, y);
                    globalHullMassive.Add(curPoint);
                }
            }
        }

        // Prints hull calculated based on globalHull
        public void PrintHullMathCoords(bool[,] field, string filename)
        {
            int[,] intField = new int[field.GetLength(1), field.GetLength(0)];
            for (int i = 0; i < field.GetLength(1); i++)
            {
                for (int j = 0; j < field.GetLength(0); j++)
                {
                    // Swapping indices for an image to look like a standard cartesian coords (up-directed y-axis)
                    intField[i, j] = field[j, i] ? 255 : 0;
                }
            }

            foreach (PointF point in globalHull)
            {
                intField[(int)Math.Round(point.Y), (int)Math.Round(point.X)] = 127;
            }

            Bitmap image = ImageHelper.SaveArrayToBitmap(intField);

            image.Save(filename);
        }

        [TestMethod]
        public void TestConvexHullBuild()
        {
            List<PointF> pointList = new List<PointF>(
                new PointF[] 
                {
                    new PointF(0, 100),
                    new PointF(200, 0),
                    new PointF(400, 200),
                    new PointF(800, 300),
                    new PointF(600, 600),
                    new PointF(300, 700),
                    new PointF(200, 600),
                    new PointF(100, 900)
                });
            
            List<PointF> resPointList = ConvexHull.GetConvexHull(pointList);

            List<PointF> resToCheckRev = new List<PointF>(globalHull);
            resToCheckRev.Reverse();

            Assert.IsTrue(resPointList.SequenceEqual(globalHull) || resPointList.SequenceEqual(resToCheckRev));
        }

        [TestMethod]
        public void TestConvexHullFieldFilling()
        {
            bool[,] field = FieldFiller.GetFieldFilling(bmpX, bmpY, globalHull);

            PrintHullMathCoords(field, "FieldFilling.jpg");
        }

        [TestMethod]
        public void TestConvexHullExtended()
        {
            List<PointF> extendedHull = ConvexHullModified.ExtendHull(globalHull, omega);

            bool[,] field = FieldFiller.GetFieldFilling(bmpX, bmpY, extendedHull);

            PrintHullMathCoords(field, "FieldFillingExtended.jpg");
        }

        [TestMethod]
        public void TestConvexHullExtendedRounded()
        {
            List<PointF> extendedHull = ConvexHullModified.ExtendHull(globalHull, omega);

            bool[,] field = ConvexHullModified.GetRoundedFieldFilling(bmpX, bmpY, omega, globalHull, extendedHull);

            PrintHullMathCoords(field, "FieldFillingExtendedRounded.jpg");
        }

        [TestMethod]
        public void TestConvexHullExtendedRoundedMassive()
        {
            string homeFolder = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            ParsePoints(homeFolder + "\\convex_hull_db.txt");

            List<PointF> hull = ConvexHull.GetConvexHull(globalHullMassive);

            List<PointF> extendedHull = ConvexHullModified.ExtendHull(hull, omega);

            bool[,] field = ConvexHullModified.GetRoundedFieldFilling(bmpX, bmpY, omega, hull, extendedHull);

            PrintHullMathCoords(field, "FieldFillingExtendedRounded.jpg");
        }
    }
}