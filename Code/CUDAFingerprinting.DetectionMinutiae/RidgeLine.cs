using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.OrientationField;

namespace CUDAFingerprinting.RidgeLine
{
	internal enum Directions
	{
		Forward,
		Back
	}

	internal enum MinutiaTypes
	{
		NotMinutia,
		LineEnding,
		Intersection
	}

	internal class RidgeOnLine
	{
		private const double Pi4 = Math.PI/4;
		private int _countTest; //for testing
		private bool _diffAngle;
		private Directions _direction;
		private readonly int[,] _image;
		private readonly int _lengthWings;

		private readonly List<Tuple<Minutia, MinutiaTypes>> _minutias;
		private readonly PixelwiseOrientationField _orientation;

		private readonly Tuple<int, int>[] _section;
		private double _sectionsAngle; //?
		private int _sectionsCenter;
		private readonly int _step;
		private readonly bool[,] _visited;

		public RidgeOnLine(int[,] image, int step, int lengthWings)
		{
			_countTest = 0;

			_image = image;
			_orientation = new PixelwiseOrientationField(image, 16); //will try change blockSize
			_step = step;
			_lengthWings = lengthWings;
			_section = new Tuple<int, int>[2*lengthWings + 1];
			_diffAngle = false;
			_visited = new bool[image.GetLength(0), image.GetLength(1)];
			_minutias = new List<Tuple<Minutia, MinutiaTypes>>();
		}

		private void AddMinutia(Minutia newMinutia, MinutiaTypes type)
		{
			_minutias.Add(new Tuple<Minutia, MinutiaTypes>(newMinutia, type));
		}

		public void FindMinutia(int x, int y, double duplicateDelta, int colorThreshold = 15)
		{
			if ((_image[y, x] >= colorThreshold) || _visited[y, x]) return;
			_visited[y, x] = true;

			FollowLine(new Tuple<int, int>(x, y), Directions.Forward);
			_diffAngle = false;


			FollowLine(new Tuple<int, int>(x, y), Directions.Back);
			_diffAngle = false;
		}

		private void NewSection(Tuple<int, int> point)
		{
			for (var index = 0; index < _section.Length; index++)
				_section[index] = new Tuple<int, int>(-1, -1);

			var x = point.Item1;
			var y = point.Item2;

			var lEnd = _lengthWings;
			var rEnd = lEnd;

			var rightE = false;
			var leftE = false;

			var angle = _orientation.GetOrientation(y, x) + Pi4*2;

			_section[_lengthWings] = point;

			for (var i = 1; i <= _lengthWings; i++)
			{
				var xs = Convert.ToInt32(x - i*Math.Cos(angle));
				var ys = Convert.ToInt32(y - i*Math.Sin(angle));
				var xe = Convert.ToInt32(x + i*Math.Cos(angle));
				var ye = Convert.ToInt32(y + i*Math.Sin(angle));

				if (!OutOfImage(xs, ys) && (_image[ys, xs] < 15) && !rightE)
				{
					_section[_lengthWings - i] = new Tuple<int, int>(xs, ys);
					rEnd--;
				}
				else
				{
					rightE = true;
				}

				if (!OutOfImage(xe, ye) && (_image[ye, xe] < 15) && !leftE)
				{
					_section[_lengthWings + i] = new Tuple<int, int>(xe, ye);
					lEnd++;
				}
				else
				{
					leftE = true;
				}

				_sectionsCenter = (lEnd + rEnd)/2;

				x = _section[_sectionsCenter].Item1;
				y = _section[_sectionsCenter].Item2;

				_sectionsAngle = _orientation.GetOrientation(y, x);
			}
		}

		private bool OutOfImage(int x, int y)
		{
			return (x < 0) || (y < 0) || (x >= _image.GetLength(1)) || (y >= _image.GetLength(0));
		}

		private Tuple<int, int> MakeStep(int _x = -1, int _y = -1, bool forPaint = false)
		{
			var x = forPaint ? _x : _section[_sectionsCenter].Item1;
			var y = forPaint ? _y : _section[_sectionsCenter].Item2;

			var __step = forPaint ? 1 : _step;

			var angle = _sectionsAngle + (int) _direction*Math.PI + (_diffAngle ? Math.PI : 0) + Math.PI*2;
			while (angle > Math.PI*2) angle -= Math.PI*2;

			var dx = x + __step*Math.Cos(angle);
			var dy = y + __step*Math.Sin(angle);

			x = (int) (dx >= 0 ? dx + 0.5 : dx - 0.5);
			y = (int) (dy >= 0 ? dy + 0.5 : dy - 0.5);

			if (OutOfImage(x, y)) return new Tuple<int, int>(-1, -1);

			var ang2Check = _orientation.GetOrientation(y, x);
			if ((_sectionsAngle*ang2Check < 0
				    ? (_sectionsAngle > 1.4) || (ang2Check > 1.4)
				    : Math.Abs(_sectionsAngle - ang2Check) > 0.5) & !forPaint)
				_diffAngle = !_diffAngle;

			return new Tuple<int, int>(x, y);
		}

		private void Paint(Tuple<int, int>[] edge, int edgesCenter)
		{
			var queue = new List<Tuple<int, int>>();
			var stopPixels = new List<Tuple<int, int>>();

			foreach (var tuple in edge.Where(tuple => tuple.Item1 != -1))
			{
				_visited[tuple.Item2, tuple.Item1] = true;
				queue.Add(tuple);
				stopPixels.Add(tuple);
			}

			foreach (var tuple in _section.Where(tuple => tuple.Item1 != -1))
			{
				_visited[tuple.Item2, tuple.Item1] = true;
				stopPixels.Add(tuple);
			}
			while (queue.Count > 0)
			{
				var x = queue[0].Item1;
				var y = queue[0].Item2;
				stopPixels.Add(queue[0]);

				queue.RemoveAt(0);

				var pCheck = MakeStep(x, y, true);

				for (var i = -1; i < 2; i++)
					for (var j = -1; j < 2; j++)
					{
						var fooPoint = new Tuple<int, int>(x + i, y + j);

						if (!OutOfImage(x + i, y + j) && (_image[y + j, x + i] < 15) && !_visited[y + j, x + i] &&
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

		private MinutiaTypes CheckStopCriteria(int threshold = 20)
		{
			var x = _section[_sectionsCenter].Item1;
			var y = _section[_sectionsCenter].Item2;

			if (_visited[y, x])
				return MinutiaTypes.Intersection;

			if (_image[y, x] > threshold)
				return MinutiaTypes.LineEnding;

			return MinutiaTypes.NotMinutia;
		}

		private void MakeTestBmp(int x, int y) //for visualisation results
		{
			var image = new int[_visited.GetLength(0), _visited.GetLength(1)];

			for (var i = 0; i < image.GetLength(1); i++)
				for (var j = 0; j < image.GetLength(0); j++)
					if (_visited[j, i])
						image[j, i] = 255;

			var bmp = ImageHelper.SaveArrayToBitmap(image);

			bmp.SetPixel(x, 364 - y, Color.Red); //What the 364??

			bmp.Save("Test" + _countTest + ".bmp");
			_countTest++;
		}

		private void FollowLine(Tuple<int, int> point, Directions direction)
		{
			NewSection(point);
			if (_section[_sectionsCenter].Item1 == -1) return;

			_direction = direction;

			MinutiaTypes minutiaType;
			int x, y;
			double angle;

			do
			{
				x = point.Item1;
				y = point.Item2;
				angle = _orientation.GetOrientation(y, x);

				var oldSection = new Tuple<int, int>[_lengthWings*2 + 1];
				for (var i = 0; i < _section.Length; i++)
					oldSection[i] = _section[i];

				var oldCenter = _sectionsCenter;

				point = MakeStep();

				angle += (int) direction*Math.PI + (_diffAngle ? Math.PI : 0);
				while (angle > Math.PI*2) angle -= Math.PI*2;

				if (point.Item1 == -1) return;

				NewSection(point);
				if (_section[_sectionsCenter].Item1 == -1) return; //?

				minutiaType = CheckStopCriteria();
				if (minutiaType == MinutiaTypes.NotMinutia) Paint(oldSection, oldCenter);
				else minutiaType = minutiaType;
			} while (minutiaType == MinutiaTypes.NotMinutia);

			x = point.Item1;
			y = point.Item2;

			var possMinutia = new Minutia();
			possMinutia.X = x;
			possMinutia.Y = y;
			possMinutia.Angle = Convert.ToSingle(angle);

			if (!IsDuplicate(possMinutia, minutiaType))
			{
				AddMinutia(possMinutia, minutiaType); //need add check of false minutias
				MakeTestBmp(x, y);
			}
		}

		private void CheckAndDeleteFalseMinutia(Minutia minutia, MinutiaTypes minutiaTypes, double delta = 3)
		{
			var i =
				_minutias.FindIndex(x => Math.Sqrt(Math.Pow(x.Item1.X - minutia.X, 2) + Math.Pow(x.Item1.Y - minutia.Y, 2)) < delta);

			_minutias.RemoveAt(i);
		}

		private bool IsDuplicate(Minutia minutia, MinutiaTypes minutiaTypes, double delta = 5)
		{
			return
				_minutias.Exists(
					x =>
						(Math.Sqrt(Math.Pow(x.Item1.X - minutia.X, 2) + Math.Pow(x.Item1.Y - minutia.Y, 2)) < delta) &&
						(x.Item2 == minutiaTypes));
		}

		public bool[,] GetVisitedMap()
		{
			return _visited;
		}

		public List<Tuple<Minutia, MinutiaTypes>> GetMinutiaList()
		{
			return _minutias;
		}
	}
}