using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common.Matrix;
using System.Drawing;

namespace CUDAFingerprinting.Common.Matrix.Tests
{
    [TestClass]
    public class MatrixTest
    {
        [TestMethod]
        public void SimpleMatrixTest()
        {
            var image = Properties.Resources.SampleFinger;
            Matrix M = new Matrix(image);

            M.SobelFilter();
            int[,] matrix = M.MatrixMaking();

            Bitmap newPic = M.BWPicture(matrix);
            newPic.Save("newPic.bmp");
        }
    }
}