using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.GPU.Segmentation.Test
{
    [TestClass]
    public class SegmentationTests
    {
        [DllImport("CUDAFingerprinting.GPU.Segmentation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "Segmentate")]
        public static extern void Segmentate(CUDAArray<float> value, int* matrix);
        [TestMethod]
        public void SegmentationTest()
        {
            float [,] fPic = ImageHelper.LoadImageAsFloat(Resources._1_8);
            int width = fPic.GetLength(0);
            int height = fPic.GetLength(1);

            float[] fPicLin;
            for (int i = 0; i < width; ++i)
            {
                for (int j = 0; j < height; ++j)
                {
                    fPicLin[i * width + j] = fPic[i, j];
                }
            }
            
            int[] matrix = new int [width * height];
            // In this matrix 1 means light shade of gray, and 0 means dark shade of gray 

            MakingMatrix(fPic, width, height, matrix);
        }
    }
}
