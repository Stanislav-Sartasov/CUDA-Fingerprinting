using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.ComTypes;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.OrientationField;

namespace CUDAFingerprinting.DetectionMinutiae
{
    class RidgeLine
    {
        private const double Pi4 = Math.PI/4;

        enum MinutiaeType {NotMinutiae, LineEnding, Intersection}
        enum Directions { Forward, Back }

	    private static int[,] _image;
	    private static bool[,] _visited;
        /*private int _lengthOfWings;*/
        private static List<Tuple<int, int>> _wings;
	    private int _step;
        private Directions _direction;
        private PixelwiseOrientationField _orientationField;
	    private List<Tuple<Minutia, MinutiaeType>> _minutiaeList;

	    private bool _diffAngle;
		private static int _mid = _wings.Count / 2;

		public RidgeLine(int[,] image, int step)
        {
	        _image = image;
			_orientationField = new PixelwiseOrientationField(image, 8);
	        _step = step;
	        _diffAngle = false;
			_visited = new bool[image.GetLength(0), image.GetLength(1)];
			_minutiaeList = new List<Tuple<Minutia, MinutiaeType>>();
        }

        private List<Tuple<int, int>> SearchNewWings(int x, int y)
        {
            /*int[] newWings = new int[_lengthOfWings * 2];*/
            double angle = _orientationField.GetOrientation(x, y) + Pi4;
            List<Tuple<int, int>> listWings = new List<Tuple<int, int>>();
	        if (OutOfRange(x, y)) return listWings;
			listWings.Add(new Tuple<int, int>(x, y));
			
	        int sx = Convert.ToInt32(x + Math.Cos(angle));
			int sy = Convert.ToInt32(y + Math.Sin(angle));

	        while (!OutOfRange(sx, sy))
	        {
				if (_image[sx, sy] > 30) break;

		        listWings.Add(new Tuple<int, int>(sx, sy));

		        sx = Convert.ToInt32(sx + Math.Cos(angle));
		        sy = Convert.ToInt32(sy + Math.Sin(angle));
	        }

	        listWings.Reverse();

			sx = Convert.ToInt32(x - Math.Cos(angle));
			sy = Convert.ToInt32(y - Math.Sin(angle));

			while (!OutOfRange(sx, sy))
			{
				if (_image[sx, sy] > 30) break;

				listWings.Add(new Tuple<int, int>(sx, sy));

				sx = Convert.ToInt32(sx - Math.Cos(angle));
				sy = Convert.ToInt32(sy - Math.Sin(angle));
			}

	        listWings.Reverse();

	        return listWings;
        }

	    private bool OutOfRange(int x, int y)
	    {
		    return x < 0 || y < 0 || _image.GetLength(0) <= x || _image.GetLength(1) <= y;
	    }

	    private Tuple<int, int> MakeStep(int px = -1, int py = -1, bool forPaint = false)
	    {
			//int mid = _wings.Count() / 2;
		    int x = forPaint ? px : _wings[_mid].Item1;
			int y = forPaint ? py : _wings[_mid].Item2;

		    int step = forPaint ? 1 : _step;
		    double wingAngle = _orientationField.GetOrientation(x, y);

			double angle = wingAngle + ((int)_direction) * Math.PI + (_diffAngle ? Math.PI : 0) + Math.PI * 2;
			while (angle > Math.PI * 2) angle -= Math.PI * 2;

			double dx = step * Math.Cos(angle);
			double dy = step * Math.Sin(angle);

			x += (int)(dx >= 0 ? dx + 0.5 : dx - 0.5);
			y += (int)(dy >= 0 ? dy + 0.5 : dy - 0.5);

		    if (OutOfRange(x, y)) return new Tuple<int, int>(-1, -1);

			if (!forPaint)
		    {
			    double ang2Check = _orientationField.GetOrientation(x, y);
			    if (((wingAngle*ang2Check < 0) ? (wingAngle > 1.4) || (ang2Check > 1.4) : Math.Abs(wingAngle - ang2Check) > 0.5))
			    {
				    _diffAngle = !_diffAngle;
			    }
		    }

		    return new Tuple<int, int>(x, y);
	    }

        private void Follow(int x, int y, Directions directions)
        {
	        List<Tuple<int, int>> dfltWings = SearchNewWings(x, y);
			if (dfltWings.Count == 0) return;

	        MinutiaeType minutiae;
	        int mx, my;
	        double angle;

	        do
	        {
		        mx = _wings[_mid].Item1;
		        my = _wings[_mid].Item2;
		        //angle = _orientationField.GetOrientation(mx, my);

		        Tuple<int, int> afterStep = MakeStep();
				if (afterStep.Item1 == -1) return;

				minutiae = TryDetectMinutiae();

		        List<Tuple<int, int>> newWings = SearchNewWings(mx, my);
				if (newWings.Count == 0) return;

				Paint(newWings);
		        _wings = newWings;
	        } while (minutiae == MinutiaeType.NotMinutiae);

			mx = _wings[_mid].Item1;
			my = _wings[_mid].Item2;
			angle = _orientationField.GetOrientation(mx, my) + ((int) directions) + (_diffAngle ? Math.PI : 0);

	        Minutia possiblyMinutia = new Minutia();

	        possiblyMinutia.X = mx;
	        possiblyMinutia.Y = my;
			possiblyMinutia.Angle = (float) angle;

	        if (!IsDuplicate(possiblyMinutia, minutiae))
	        {
		        AddMinutiae(possiblyMinutia, minutiae);
	        }
        }

	    private void AddMinutiae(Minutia newMinutiae, MinutiaeType type)
	    {
			_minutiaeList.Add(new Tuple<Minutia, MinutiaeType>(newMinutiae, type));
		}

	    private void Paint(List<Tuple<int, int>> finish)
	    {
		    List<Tuple<int, int>> queue = new List<Tuple<int, int>>();
		    List<Tuple<int, int>> stopPixels = new List<Tuple<int, int>>();

		    foreach (var tuple in finish.Where(tuple => tuple.Item1 != -1))
		    {
			    _visited[tuple.Item2, tuple.Item1] = true;
			    queue.Add(tuple);
			    stopPixels.Add(tuple);
		    }

		    foreach (var tuple in _wings.Where(tuple => tuple.Item1 != -1))
		    {
			    _visited[tuple.Item2, tuple.Item1] = true;
			    stopPixels.Add(tuple);
		    }
		    while (queue.Count > 0)
		    {
			    int x = queue[0].Item1;
			    int y = queue[0].Item2;
			    stopPixels.Add(queue[0]);

			    queue.RemoveAt(0);

			    var pCheck = MakeStep(x, y, true);

			    for (int i = -1; i < 2; i++)
			    {
				    for (int j = -1; j < 2; j++)
				    {
					    var fooPoint = new Tuple<int, int>(x + i, y + j);

					    if (!OutOfRange(x + i, y + j) && _image[y + j, x + i] < 15 && !_visited[y + j, x + i] &&
					        !queue.Exists(q => (q.Item1 == fooPoint.Item1)
					                           && (q.Item2 == fooPoint.Item2)) &&
					        !stopPixels.Exists(q => (q.Item1 == fooPoint.Item1) && (q.Item2 == fooPoint.Item2)))
					    {
						    queue.Add(fooPoint);
						    _visited[y + j, x + i] = true;
					    }
				    }
			    }

		    }
	    }

	    static MinutiaeType TryDetectMinutiae(int threshold = 40)
	    {
		    int x = _wings[_mid].Item1;
		    int y = _wings[_mid].Item2;

		    if (_visited[y, x])
		    {
			    return MinutiaeType.Intersection;
		    }

		    if (_image[y, x] > threshold)
		    {
			    return MinutiaeType.LineEnding;
		    }

			return MinutiaeType.NotMinutiae;
	    }

		private bool IsDuplicate(Minutia minutia, MinutiaeType minutiaTypes, double delta = 5)
		{
			return
				_minutiaeList.Exists(
					x => Math.Sqrt(Math.Pow(x.Item1.X - minutia.X, 2) + Math.Pow(x.Item1.Y - minutia.Y, 2)) < delta && x.Item2 == minutiaTypes);
		}

	    static void Main(string[] args)
	    {
		    
	    }
	}
}
