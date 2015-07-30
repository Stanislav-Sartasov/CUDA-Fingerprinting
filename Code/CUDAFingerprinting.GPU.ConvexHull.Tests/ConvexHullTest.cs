using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting;
using CUDAFingerprinting.Common.ConvexHull.Tests;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.InteropServices;

namespace CUDAFingerprinting.GPU.ConvexHull.Tests
{
    public struct Point
    {
        float X;
        float Y;
    }

    [TestClass]
    public class ConvexHullTest
    {
        [DllImport("CUDAFingerprinting.GPU.ConvexHull.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "extendHull")]
        public static extern IntPtr extendHull(IntPtr hull, int hullLength, float omega);

        [DllImport("CUDAFingerprinting.GPU.ConvexHull.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "getRoundFieldFilling")]
        public static extern IntPtr extendHull(
            int rows, int columns, float omega, IntPtr hull, int hullLength, IntPtr extendedHull, int extendedHullLength);

        [DllImport("CUDAFingerprinting.GPU.ConvexHull.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "getRoundFieldFilling")]
        public static extern void printHullMathCoords(IntPtr field, IntPtr filename);

        [TestMethod]
        public void TestConvexHullExtendedRoundedMassive()
        {
            string homeFolder = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            ConvexHullTests.ParsePoints(homeFolder + "\\convex_hull_db.txt");

            int cylinderSize = Marshal.SizeOf(typeof(Point));

            List<PointF> hull = Common.ConvexHull.ConvexHull.GetConvexHull(ConvexHullTests.globalHullMassive);

            List<PointF> extendedHull = Common.ConvexHull.ConvexHullModified.ExtendHull(hull, ConvexHullTests.omega);            
        }
    }
}
