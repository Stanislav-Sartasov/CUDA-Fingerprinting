using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace CUDAFingerprinting.Common.ConvexHull.Tests
{
    [TestClass]
    public class ConvexHullTests
    {

        List<PointF> globalHullToCheck = new List<PointF>(
            new PointF[]
            {
                new PointF(0, 1),
                new PointF(2, 0),
                new PointF(8, 3),
                new PointF(6, 6),
                new PointF(1, 9),
            });

        List<PointF> globalHull = new List<PointF>(
            new PointF[]
            {
                new PointF(100, 100),
                new PointF(400, 100),
                new PointF(400, 1000),
                new PointF(100, 1000)
            });
        
        // Prints hull calculated based on globalHull
        public void PrintHull(bool[,] field, string filename)
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
        public void ConvexHullBuildTest()
        {
            List<PointF> pointList = new List<PointF>(
                new PointF[] 
                {
                    new PointF(0, 1),
                    new PointF(2, 0),
                    new PointF(4, 2),
                    new PointF(8, 3),
                    new PointF(6, 6),
                    new PointF(3, 7),
                    new PointF(2, 6),
                    new PointF(1, 9)
                });

            List<PointF> resPointList = ConvexHull.GetConvexHull(pointList);

            List<PointF> resToCheckRev = new List<PointF>(globalHullToCheck);
            resToCheckRev.Reverse();

            Assert.IsTrue(resPointList.SequenceEqual(globalHullToCheck) || resPointList.SequenceEqual(resToCheckRev));
        }

        [TestMethod]
        public void ConvexHullFieldFillingTest()
        {
            bool[,] field = FieldFiller.GetFieldFilling(500, 1100, globalHull);

            PrintHull(field, "FieldFilling.jpg");
        }

        [TestMethod]
        public void ConvexHullExtendedTest()
        {
            double omega = 40;

            List<PointF> extendedHull = ConvexHullModified.ExtendHull(globalHull, omega);

            bool[,] field = FieldFiller.GetFieldFilling(1100, 1100, extendedHull);

            PrintHull(field, "FieldFillingExtended.jpg");
        }

        [TestMethod]
        public void ConvexHullExtendedRoundedTest()
        {
            double omega = 40;

            List<PointF> extendedHull = ConvexHullModified.ExtendHull(globalHull, omega);

            bool[,] field = ConvexHullModified.GetRoundedFieldFilling(1100, 1100, 40, globalHull, extendedHull);

            PrintHull(field, "FieldFillingExtendedRounded.jpg");
        }
    }
}