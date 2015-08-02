using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.Common.Tests
{
    [TestClass]
    public class ConvolutionTests
    {
        [TestMethod]
        public void TestConvolutionForEvenSizedFilters()
        {
            var img = ImageHelper.LoadImage<double>(Resources.SampleFinger);

            var kernel = KernelHelper.MakeKernel((x, y) => 1, 4);

            var result = ConvolutionHelper.Convolve(img, kernel);

            ImageHelper.SaveArrayAndOpen(result);
        }
    }
}
