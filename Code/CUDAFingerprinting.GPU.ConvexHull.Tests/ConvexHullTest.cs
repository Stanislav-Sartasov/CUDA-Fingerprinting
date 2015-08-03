using System;
using CUDAFingerprinting.Common.ConvexHull.Tests;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Runtime.InteropServices;

namespace CUDAFingerprinting.GPU.ConvexHull.Tests
{
    public struct Point
    {
        public float X;
        public float Y;
    }

    // For now fails due to last Marshal.Copy (how to handle C boolean arrays?)
    [TestClass]
    public class ConvexHullTest
    {
        [DllImport("CUDAFingerprinting.GPU.ConvexHull.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "initConvexHull")]
        public static extern void initConvexHull(int givenFieldHeight, int givenFieldWidth, int givenMaxPointCount);

        [DllImport("CUDAFingerprinting.GPU.ConvexHull.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "processConvexHull")]
        public static extern IntPtr processConvexHull(IntPtr points, float omega, int actualPointCount);

        [DllImport("CUDAFingerprinting.GPU.ConvexHull.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "terminateConvexHull")]
        public static extern void terminateConvexHull();

        public static int testFieldHeight = 1100;
        public static int testFieldWidth = 1100;

        public static int testMaxPointCount = 1100;

        public static float testOmega = 40;

        [TestMethod]
        public void TestConvexHullExtendedRoundedMassive()
        {
            string homeFolder = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            ConvexHullTests.ParsePoints(homeFolder + "\\convex_hull_db.txt");

            int pointSize = Marshal.SizeOf(typeof(Point));

            IntPtr pointsUnmanaged = Marshal.AllocHGlobal(ConvexHullTests.globalHullMassive.Count * pointSize);
            IntPtr curHullPtr = new IntPtr(pointsUnmanaged.ToInt32()); // No idea why not just " = db", copypasted from SO

            for (int i = 0; i < ConvexHullTests.globalHullMassive.Count; i++)
            {
                Point curPoint = new Point();
                curPoint.X = ConvexHullTests.globalHullMassive[i].X;
                curPoint.Y = ConvexHullTests.globalHullMassive[i].Y;

                Marshal.StructureToPtr(curPoint, curHullPtr, false);
                curHullPtr = new IntPtr(curHullPtr.ToInt32() + Marshal.SizeOf(typeof(Point)));
            }

            initConvexHull(testFieldHeight, testFieldWidth, testMaxPointCount);
            IntPtr fieldPtr = processConvexHull(pointsUnmanaged, testOmega, ConvexHullTests.globalHullMassive.Count);
            //Console.WriteLine(fieldPtr);

            byte[] field = new byte[testFieldHeight * testFieldWidth];

            Marshal.Copy(fieldPtr, field, 0, testFieldHeight * testFieldWidth);

            int[,] intField = new int[testFieldHeight, testFieldWidth];
            for (int i = 0; i < testFieldHeight * testFieldWidth; i++)
            {
                intField[i % testFieldWidth, i / testFieldWidth] = field[i] != 0 ? 255 : 0;
            }

            Common.ImageHelper.SaveArray(intField, "TestFieldFillingExtendedRounded.jpg");

            terminateConvexHull();
        }
    }
}
