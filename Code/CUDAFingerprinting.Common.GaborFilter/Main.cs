using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.Common.GaborFilter
{
    class Program
    {
        private static void Main(string[] args)
        {
            var mas = new GaborFilter(8, 5);

            mas.Filters[2].WriteMatrix();

            Console.ReadKey();
        }
    }
}
