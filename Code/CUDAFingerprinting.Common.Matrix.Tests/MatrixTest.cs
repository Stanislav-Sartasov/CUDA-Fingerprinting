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
            Bitmap SFPic = M.SobelFilter();

            Matrix M2 = new Matrix(SFPic);

            M2.MatrixMaking();
            Bitmap newPic = M2.BWPicture();
            newPic.Save("newPic.bmp");
        }
    }
}
