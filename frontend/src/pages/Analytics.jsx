import { useState } from "react";
import Calendar from "react-calendar";
import "react-calendar/dist/Calendar.css";
import Navbar from "../components/Navbar";
import backgroundImg from "../assets/background.png";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import "./calendar.css";

const rangeSample = [
  { "date": "2025-02-10-05:49", "value": 16 },
  { "date": "2025-02-11-20:28", "value": 26 },
  { "date": "2025-02-12-16:02", "value": 48 },
  { "date": "2025-02-13-07:46", "value": 16 },
  { "date": "2025-02-14-13:43", "value": 20 },
  { "date": "2025-02-15-04:22", "value": 8 },
  { "date": "2025-02-16-18:06", "value": 37 },
  { "date": "2025-02-17-01:20", "value": 42 },
  { "date": "2025-02-18-14:15", "value": 40 },
  { "date": "2025-02-19-22:24", "value": 15 },
  { "date": "2025-02-20-09:05", "value": 11 },
  { "date": "2025-02-21-07:56", "value": 49 },
  { "date": "2025-02-22-06:20", "value": 13 },
  { "date": "2025-02-23-23:36", "value": 19 },
  { "date": "2025-02-24-15:50", "value": 10 },
  { "date": "2025-02-25-19:38", "value": 35 },
  { "date": "2025-02-26-11:15", "value": 29 },
  { "date": "2025-02-27-03:03", "value": 7 },
  { "date": "2025-02-28-17:26", "value": 47 },
  { "date": "2025-03-01-20:19", "value": 14 },
  { "date": "2025-03-02-10:31", "value": 23 },
  { "date": "2025-03-03-01:14", "value": 44 },
  { "date": "2025-03-04-05:42", "value": 33 },
  { "date": "2025-03-05-08:00", "value": 30 },
  { "date": "2025-03-06-22:51", "value": 21 },
  { "date": "2025-03-07-12:48", "value": 46 },
  { "date": "2025-03-08-16:33", "value": 28 },
  { "date": "2025-03-09-00:55", "value": 50 },
  { "date": "2025-03-10-04:18", "value": 25 },
  { "date": "2025-03-11-09:09", "value": 36 },
  { "date": "2025-03-12-13:17", "value": 22 },
  { "date": "2025-03-13-17:40", "value": 18 },
  { "date": "2025-03-14-14:12", "value": 27 },
  { "date": "2025-03-15-06:34", "value": 9 },
  { "date": "2025-03-16-11:46", "value": 41 },
  { "date": "2025-03-17-15:39", "value": 45 },
  { "date": "2025-03-18-19:03", "value": 34 },
  { "date": "2025-03-19-02:51", "value": 38 },
  { "date": "2025-03-20-22:16", "value": 12 },
  { "date": "2025-03-21-07:07", "value": 6 },
  { "date": "2025-03-22-00:27", "value": 39 },
  { "date": "2025-03-23-03:19", "value": 24 },
  { "date": "2025-03-24-21:11", "value": 31 },
  { "date": "2025-03-25-10:22", "value": 43 },
  { "date": "2025-03-26-05:58", "value": 35 },
  { "date": "2025-03-27-17:46", "value": 17 },
  { "date": "2025-03-28-09:30", "value": 20 },
  { "date": "2025-03-29-18:38", "value": 15 },
  { "date": "2025-03-30-06:43", "value": 48 },
  { "date": "2025-03-31-01:09", "value": 19 },
  { "date": "2025-04-01-23:59", "value": 44 },
  { "date": "2025-04-02-07:27", "value": 26 },
  { "date": "2025-04-03-12:01", "value": 40 },
  { "date": "2025-04-04-15:46", "value": 32 },
  { "date": "2025-04-05-04:14", "value": 38 },
  { "date": "2025-04-06-13:21", "value": 7 },
  { "date": "2025-04-07-19:55", "value": 28 },
  { "date": "2025-04-08-02:03", "value": 22 },
  { "date": "2025-04-09-08:30", "value": 46 },
  { "date": "2025-04-10-16:20", "value": 11 }
]
;

const getMonthName = (monthIndex) =>
  [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ][monthIndex];

const getSeason = (month) => {
  if ([11, 0, 1].includes(month)) return "Winter";
  if ([2, 3, 4].includes(month)) return "Spring";
  if ([5, 6, 7].includes(month)) return "Summer";
  return "Autumn";
};

const groupData = (type, rangeData) => {
  const grouped = {};

  rangeData.forEach((item) => {
    const itemDate = new Date(item.date.replace(/-(\d\d:\d\d)$/, "T$1"));

    let key = "";
    switch (type) {
      case "byDay":
        key = itemDate.toLocaleDateString("en-US", { weekday: "long" });
        break;
      case "byHour":
        key = itemDate.toTimeString().slice(0, 5);
        break;
      case "byMonth":
        key = getMonthName(itemDate.getMonth());
        break;
      case "bySeason":
        key = getSeason(itemDate.getMonth());
        break;
      default:
        return;
    }

    if (!grouped[key]) grouped[key] = 0;
    grouped[key] += item.value;
  });

  return Object.entries(grouped).map(([name, value]) => ({ name, value }));
};

export default function Profile() {
  const [calendarDate, setCalendarDate] = useState(new Date());
  const [chartType, setChartType] = useState("byDay");
  const [chartStyle, setChartStyle] = useState("bar");
  const [rangeEnabled, setRangeEnabled] = useState(false);
  const [dateRange, setDateRange] = useState([new Date(), new Date()]);

  const getChartData = () => {
    if (chartType === "range") {
      const [start, end] = dateRange;
      const filtered = rangeSample.filter((item) => {
        const d = new Date(item.date.replace(/-(\d\d:\d\d)$/, "T$1"));
        return d >= start && d <= end;
      });

      return groupData("byDay", filtered);
    }

    return groupData(chartType, rangeSample);
  };

  const data = getChartData();

  const renderChart = () => {
    if (chartStyle === "line") {
      return (
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Legend />
          <Line dataKey="value" stroke="#fb923c" strokeWidth={3} />
        </LineChart>
      );
    } else if (chartStyle === "area") {
      return (
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorUv" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#fb923c" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#fb923c" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Legend />
          <Area
            dataKey="value"
            stroke="#fb923c"
            fillOpacity={1}
            fill="url(#colorUv)"
          />
        </AreaChart>
      );
    }

    return (
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" />
        <YAxis allowDecimals={false} />
        <Tooltip />
        <Legend />
        <Bar dataKey="value" fill="#fb923c" radius={[10, 10, 0, 0]} />
      </BarChart>
    );
  };

  return (
    <>
      <Navbar />
      <div
        className="min-h-screen flex items-center justify-center bg-cover bg-no-repeat bg-center px-4"
        style={{ backgroundImage: `url(${backgroundImg})` }}
      >
        <div className="w-full max-w-4xl bg-white rounded-3xl shadow-xl p-6">
          <h1 className="text-2xl font-bold mb-6 text-center text-orange-600">
            Analytics Dashboard
          </h1>

          <div className="flex flex-col md:flex-row gap-6 mb-6">
            <div className="w-full md:w-1/2">
              <h2 className="text-lg font-medium mb-2 text-gray-800">
                Calendar
              </h2>
              <div className="scale-90 origin-top-left">
                <Calendar
                  onChange={(value) => {
                    if (!rangeEnabled) setCalendarDate(value);
                    else setDateRange(value);
                  }}
                  selectRange={rangeEnabled}
                  value={rangeEnabled ? dateRange : calendarDate}
                  className="custom-calendar"
                />
              </div>
              {rangeEnabled && chartType === "range" && (
                <p className="text-sm text-gray-500 mt-2">
                  Selected range: {dateRange[0].toLocaleDateString()} -{" "}
                  {dateRange[1].toLocaleDateString()}
                </p>
              )}
            </div>

            <div className="w-full md:w-1/2 flex flex-col gap-4">
              <div>
                <label className="text-sm font-medium text-gray-700">
                  Select Chart Type
                </label>
                <select
                  value={chartType}
                  onChange={(e) => {
                    const val = e.target.value;
                    setChartType(val);
                    setRangeEnabled(val === "range");
                  }}
                  className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring focus:border-orange-300"
                >
                  <option value="byDay">By Day</option>
                  <option value="byHour">By Hour</option>
                  <option value="byMonth">By Month</option>
                  <option value="bySeason">By Season</option>
                  <option value="range">Select Date Range</option>
                </select>
              </div>
              <div>
                <label className="text-sm font-medium text-gray-700">
                  Select Chart Style
                </label>
                <select
                  value={chartStyle}
                  onChange={(e) => setChartStyle(e.target.value)}
                  className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring focus:border-orange-300"
                >
                  <option value="bar">Bar</option>
                  <option value="line">Line</option>
                  <option value="area">Area</option>
                </select>
              </div>
            </div>
          </div>

          {data.length === 0 ? (
            <div className="w-full bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 rounded-lg">
              <p className="font-medium">Данных за этот период нет.</p>
            </div>
          ) : (
            <div className="w-full h-[400px] mt-6">
              <ResponsiveContainer width="100%" height="100%">
                {renderChart()}
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
