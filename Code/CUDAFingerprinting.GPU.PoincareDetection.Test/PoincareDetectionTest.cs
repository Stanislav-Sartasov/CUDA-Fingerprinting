using System;
using System.IO;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.OrientationField;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.GPU.PoincareDetection.Test
{
    [TestClass]
    public class PoincareDetectionTest
    {
        [DllImport("CUDAFingerprinting.GPU.PoincareDetection.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "PoincareDetect")]
        public static extern  void PoincareDetect(int[] src, int width, int height, float[] oriented, int[]dist);


        [TestMethod]
        public void PoincareDetectionTestMethod()
        {

            var img = ImageHelper.LoadImage<int>(Resource1._44_8);
            int imgWidth = img.GetLength(0), imgHeight = img.GetLength(1);

            PixelwiseOrientationField img2 = new PixelwiseOrientationField(img, 16);
            var oriented = img2.Orientation;

            int[] distPtr = new int[imgWidth*imgHeight];

            float[] orientedPtr = oriented.Select2D(x => (float) x).Make1D();

            int[] imgPtr = img.Make1D();

            PoincareDetect(imgPtr, imgHeight, imgWidth, orientedPtr, distPtr);

            var dist = distPtr.Make2D(imgWidth, imgHeight);
            ImageHelper.SaveArrayAndOpen(dist.Select2D(x=>(double)x),Path.GetTempPath() + Guid.NewGuid() + ".bmp");
        }
    }
}
