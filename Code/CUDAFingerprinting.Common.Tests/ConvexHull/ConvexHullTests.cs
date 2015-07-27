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

            List<PointF> resToCheck = new List<PointF>(
                new PointF[]
                {
                    new PointF(1, 9),
                    new PointF(6, 6),
                    new PointF(8, 3),
                    new PointF(2, 0),
                    new PointF(0, 1)
                });

            List<PointF> resToCheckRev = new List<PointF>(resToCheck);
            resToCheckRev.Reverse();

            Assert.IsTrue(resPointList.SequenceEqual(resToCheck) || resPointList.SequenceEqual(resToCheckRev));
        }

        [TestMethod]
        public void ConvexHullFieldFillingTest()
        {
            List<PointF> hull = new List<PointF>(
                new PointF[]
                {
                    new PointF(1, 9),
                    new PointF(6, 6),
                    new PointF(8, 3),
                    new PointF(2, 0),
                    new PointF(0, 1)
                });

            bool[,] field = FieldFiller.GetFieldFilling(20, 20, hull);

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

        [TestMethod]
        public void ConvexHullExtendedTest()
        {
            List<PointF> hull = new List<PointF>(
                new PointF[]
                {
                    new PointF(0, 1),
                    new PointF(2, 0),
                    new PointF(8, 3),
                    new PointF(6, 6),
                    new PointF(1, 9),
                });

            double omega = 3;

            List<PointF> extendedHull = ConvexHullMCCExtension.ExtendHull(hull, omega);

            bool[,] field = FieldFiller.GetFieldFilling(20, 20, extendedHull);

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

            image.Save("FieldFillingExtended.jpg");
        }
    }
}