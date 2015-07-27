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
        [TestMethod]
        public void ConvexHullBuildTest()
        {
            List<Point> pointList = new List<Point>(
                new Point[] 
                {
                    new Point(0, 1),
                    new Point(2, 0),
                    new Point(4, 2),
                    new Point(8, 3),
                    new Point(6, 6),
                    new Point(3, 7),
                    new Point(2, 6),
                    new Point(1, 9)
                });

            List<Point> resPointList = ConvexHull.GetConvexHull(pointList);

            List<Point> resToCheck = new List<Point>(
                new Point[]
                {
                    new Point(1, 9),
                    new Point(6, 6),
                    new Point(8, 3),
                    new Point(2, 0),
                    new Point(0, 1)
                });

            List<Point> resToCheckRev = new List<Point>(resToCheck);
            resToCheckRev.Reverse();

            Assert.IsTrue(resPointList.SequenceEqual(resToCheck) || resPointList.SequenceEqual(resToCheckRev));
        }

        [TestMethod]
        public void ConvexHullFieldFillingTest()
        {
            List<Point> hull = new List<Point>(
                new Point[]
                {
                    new Point(1, 9),
                    new Point(6, 6),
                    new Point(8, 3),
                    new Point(2, 0),
                    new Point(0, 1)
                });

            bool[,] field = FieldFiller.GetFieldFilling(10, 10, hull);

            int[,] intField = new int[field.GetLength(0), field.GetLength(1)];
            for (int i = 0; i < field.GetLength(0); i++)
            {
                for (int j = 0; j < field.GetLength(1); j++)
                {
                    // Swapping indices for an image to look like a standard cartesian coords (up-directed y-axis)
                    intField[i, j] = field[j, i] ? 255 : 0;
                }
            }

            Bitmap image = ImageHelper.SaveArrayToBitmap(intField);

            image.Save("FieldFilling.jpg");
        }
    }
}