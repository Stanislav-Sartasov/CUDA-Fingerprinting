using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.ImageProcessing.Triangulation;

namespace CUDAFingerprinting.ImageProcessing.Tests
{
    [TestClass]
    public class DelaunayTest
    {
        [TestMethod]
        public void TestTriangulation()
        {
            List<DelaunayTriangulation.Point> points = new List<DelaunayTriangulation.Point>();
            points.Add(new DelaunayTriangulation.Point(0,0));
            points.Add(new DelaunayTriangulation.Point(0,2));
            points.Add(new DelaunayTriangulation.Point(0,-2));
            points.Add(new DelaunayTriangulation.Point(2,1));
            points.Add(new DelaunayTriangulation.Point(2,-1));
            points.Add(new DelaunayTriangulation.Point(-2,1));
            points.Add(new DelaunayTriangulation.Point(-2,-1));

            List<DelaunayTriangulation.Triangle> triangles = DelaunayTriangulation.CreateDelaunayTriangulation(points);
        }
    }
}
