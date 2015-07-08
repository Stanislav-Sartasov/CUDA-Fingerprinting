using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Runtime.InteropServices;
using  CUDAFingerprinting.Common;
using System.IO;

namespace CUDAFingerprinting.GPU.ImageBinarization.CUDATest
{
    [TestClass]
    public class ImageBinarizationTest
    {
        [DllImport("CUDAFingerprinting.GPU.ImageBinarization.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "BinarizateImage")]
        public static extern void BinarizateImage(int line, int[,] dist, int width, int height);
        [TestMethod]
        public void BinarizationTest()
        {
            int[,] image = ImageHelper.LoadImageAsInt(Resources._2_6);
            int x = image.GetLength(0);
            int y = image.GetLength(1);
            BinarizateImage(128, image, x, y);
            ImageHelper.SaveArrayToBitmap(image).Save(Path.GetTempPath() + Guid.NewGuid() + ".bmp");

        }
    }
}
