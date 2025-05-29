'use client';

import React, { useEffect, useRef, useState } from "react";
import {
    createChart,
    IChartApi,
    ISeriesApi,
    CandlestickData,
    SeriesMarker,
    Time,
    LineData,
    IPriceLine,
    LineStyle
} from "lightweight-charts";
import axios from "axios";
import { Combobox } from '@headlessui/react';
import TimelineRangeSelector from './TimelineRangeSelector';
import AdvancedSettings, { PatternDetectionSettings } from './AdvancedSettings';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// Common stock symbols
const COMMON_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
    'JNJ', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'XOM', 'DIS', 'NFLX', 'ADBE'
];

// Pattern marker colors
const PATTERN_COLORS = [
    '#2196F3', // Blue
    '#4CAF50', // Green
    '#FFC107', // Amber
    '#9C27B0', // Purple
    '#FF5722', // Deep Orange
    '#00BCD4', // Cyan
    '#FF9800', // Orange
    '#E91E63', // Pink
];

interface Pattern {
    "Chart Pattern": string;
    Start: string;
    End: string;
    Seg_Start: string;
    Seg_End: string;
    Calc_Start: string;
    Calc_End: string;
}

interface OHLCData {
    Date: string;
    Open: number;
    High: number;
    Low: number;
    Close: number;
    Volume: number;
}

interface ChartMarker extends SeriesMarker<Time> {
    position: 'aboveBar' | 'belowBar';
    color: string;
    shape: 'arrowUp' | 'arrowDown';
    text: string;
}

const ChartContainer: React.FC = () => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candlestickSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const patternSeriesRef = useRef<ISeriesApi<"Line">[]>([]);
    const priceLinesRef = useRef<IPriceLine[]>([]);

    const [symbol, setSymbol] = useState<string>("AAPL");
    const [query, setQuery] = useState('');
    const [loading, setLoading] = useState<boolean>(false);
    const [patterns, setPatterns] = useState<Pattern[]>([]);
    const [error, setError] = useState<string>("");
    const [allData, setAllData] = useState<OHLCData[]>([]);
    const [visibleData, setVisibleData] = useState<OHLCData[]>([]);
    const [visibleRange, setVisibleRange] = useState<{ from: Date; to: Date }>(() => {
        const endDate = new Date();
        const startDate = new Date();
        startDate.setFullYear(startDate.getFullYear() - 1); // One year ago
        return { from: startDate, to: endDate };
    });
    const [advancedSettings, setAdvancedSettings] = useState<PatternDetectionSettings>({
        min_window_size: 10,
        max_window_size: 50,
        padding_proportion: 0.6,
        stride: 1,
        prob_threshold_of_no_pattern: 0.5,
        min_eps: 0.02,
        max_eps: 0.1,
        min_samples: 3,
        iou_threshold: 0.4,
        probability_difference_threshold: 0.1,
        final_avg_probability_threshold: 0.85,
        pattern_thresholds: {
            double_top: 0.9,
            double_bottom: 0.9,
            triangle_symmetrical: 0.4309,
            head_and_shoulders_top: 0.4583,
            head_and_shoulders_bottom: 0.3961,
            flag_high_and_tight: 0.5506
        }
    });
    const [detectionProgress, setDetectionProgress] = useState<number>(0);
    const [detectionStatus, setDetectionStatus] = useState<string>('idle');

    // Filter symbols based on search query
    const filteredSymbols = query === ''
        ? COMMON_SYMBOLS
        : COMMON_SYMBOLS.filter((symbol) =>
            symbol.toLowerCase().includes(query.toLowerCase())
        );

    const handleSymbolChange = (value: string | null) => {
        if (value) {
            setSymbol(value);
            loadAllData(value);
        }
    };

    const handleSettingsChange = (settings: PatternDetectionSettings) => {
        setAdvancedSettings(settings);
    };

    const zoomToPattern = (start: string, end: string) => {
        if (!chartRef.current) return;

        const from = Math.floor(new Date(start).getTime() / 1000) as Time;
        const to = Math.floor(new Date(end).getTime() / 1000) as Time;

        chartRef.current.timeScale().setVisibleRange({ from, to });
    };

    const loadAllData = async (selectedSymbol: string) => {
        setLoading(true);
        setError("");

        try {
            // Get data from 5 years ago to now
            const endDate = new Date();
            const startDate = new Date();
            startDate.setFullYear(startDate.getFullYear() - 5);

            const response = await axios.get<OHLCData[]>(`${API_BASE_URL}/ohlc-data/`, {
                params: {
                    symbol: selectedSymbol,
                    start_date: startDate.toISOString().split('T')[0],
                    end_date: endDate.toISOString().split('T')[0],
                },
            });

            const sortedData = response.data.sort((a, b) =>
                new Date(a.Date).getTime() - new Date(b.Date).getTime()
            );

            setAllData(sortedData);
            setVisibleData(sortedData);

            // Update visible range based on available data
            if (sortedData.length > 0) {
                const firstDate = new Date(sortedData[0].Date);
                const lastDate = new Date(sortedData[sortedData.length - 1].Date);
                setVisibleRange({ from: firstDate, to: lastDate });
            }

            if (candlestickSeriesRef.current) {
                const chartData: CandlestickData[] = sortedData.map((item: OHLCData) => ({
                    time: item.Date,
                    open: parseFloat(item.Open.toString()),
                    high: parseFloat(item.High.toString()),
                    low: parseFloat(item.Low.toString()),
                    close: parseFloat(item.Close.toString()),
                }));

                candlestickSeriesRef.current.setData(chartData);
                if (chartRef.current) {
                    chartRef.current.timeScale().fitContent();
                }
            }
        } catch (err) {
            console.error("Error fetching data:", err);
            setError(err instanceof Error ? err.message : "Error fetching data");
        } finally {
            setLoading(false);
        }
    };

    const clearPatternVisualizations = () => {
        // Remove existing pattern line series
        patternSeriesRef.current.forEach(series => {
            if (chartRef.current) {
                chartRef.current.removeSeries(series);
            }
        });
        patternSeriesRef.current = [];

        // Remove existing price lines
        priceLinesRef.current.forEach(priceLine => {
            if (candlestickSeriesRef.current) {
                candlestickSeriesRef.current.removePriceLine(priceLine);
            }
        });
        priceLinesRef.current = [];
    };

    const getPatternPoints = (pattern: Pattern, data: OHLCData[]) => {
        const startTime = new Date(pattern.Start).getTime();
        const endTime = new Date(pattern.End).getTime();

        // Find data points within the pattern range
        const patternData = data.filter(item => {
            const itemTime = new Date(item.Date).getTime();
            return itemTime >= startTime && itemTime <= endTime;
        });

        if (patternData.length === 0) return [];

        const patternType = pattern["Chart Pattern"].toLowerCase();

        // Calculate key points based on pattern type
        switch (true) {
            case patternType.includes('double top'):
                return getDoubleTopPoints(patternData, startTime, endTime);
            case patternType.includes('double bottom'):
                return getDoubleBottomPoints(patternData, startTime, endTime);
            case patternType.includes('Head-and-shoulders top'):
                return getHeadAndShouldersTopPoints(patternData, startTime, endTime);
            case patternType.includes('Head-and-shoulders bottom'):
                return getHeadAndShouldersBottomPoints(patternData, startTime, endTime);
            case patternType.includes('Triangle, symmetrical'):
                return getTrianglePoints(patternData, startTime, endTime);
            case patternType.includes('Flag, high and tight'):
                return getFlagPoints(patternData, startTime, endTime);
            default:
                return getTrendLinePoints(patternData, startTime, endTime);
        }
    };

    const getDoubleTopPoints = (data: OHLCData[], startTime: number, endTime: number): { time: Time; value: number }[] => {
        if (data.length < 3) return [];

        // Filter data within the time range
        const rangeData = data.filter(d => {
            const t = new Date(d.Date).getTime();
            return t >= startTime && t <= endTime;
        });

        type PatternPoints = {
            peak1: OHLCData;
            valley: OHLCData;
            peak2: OHLCData;
        };

        let bestPattern: PatternPoints | null = null;
        let bestScore = -Infinity;

        for (let i = 1; i < rangeData.length - 1; i++) {
            const prev = rangeData[i - 1];
            const curr = rangeData[i];
            const next = rangeData[i + 1];

            // Look for a valley flanked by two peaks (Double Top)
            if (curr.Low < prev.Low && curr.Low < next.Low) {
                const leftCandidates = rangeData.slice(0, i).filter(p => p.High > curr.Low);
                const rightCandidates = rangeData.slice(i + 1).filter(p => p.High > curr.Low);

                leftCandidates.forEach(peak1 => {
                    rightCandidates.forEach(peak2 => {
                        const timeDiff = Math.abs(new Date(peak2.Date).getTime() - new Date(peak1.Date).getTime());

                        if (timeDiff > 3 * 24 * 60 * 60 * 1000) { // at least 3 days apart
                            // Score based on similarity of peaks and depth of valley
                            const peakAvg = (peak1.High + peak2.High) / 2;
                            const valleyDepth = peakAvg - curr.Low;
                            const peakDiff = Math.abs(peak1.High - peak2.High);

                            const score = -peakDiff + valleyDepth; // Prefer symmetrical, deep valleys
                            if (score > bestScore) {
                                bestScore = score;
                                bestPattern = { peak1, valley: curr, peak2 };
                            }
                        }
                    });
                });
            }
        }

        if (!bestPattern) return [];

        // Type guard to ensure bestPattern is not null
        const pattern = bestPattern as PatternPoints;

        const result: { time: Time; value: number }[] = [
            { time: pattern.peak1.Date as Time, value: pattern.peak1.High },
            { time: pattern.valley.Date as Time, value: pattern.valley.Low },
            { time: pattern.peak2.Date as Time, value: pattern.peak2.High }
        ];

        return result;
    };


    const getDoubleBottomPoints = (data: OHLCData[], startTime: number, endTime: number) => {
        if (data.length < 3) return [];

        // Find the two lowest points
        const sortedByLow = [...data].sort((a, b) => a.Low - b.Low);
        const firstTrough = sortedByLow[0];
        const secondTrough = sortedByLow.find(item =>
            Math.abs(new Date(item.Date).getTime() - new Date(firstTrough.Date).getTime()) > 7 * 24 * 60 * 60 * 1000
        ) || sortedByLow[1];

        // Find the peak between troughs
        const firstTroughTime = new Date(firstTrough.Date).getTime();
        const secondTroughTime = new Date(secondTrough.Date).getTime();
        const minTime = Math.min(firstTroughTime, secondTroughTime);
        const maxTime = Math.max(firstTroughTime, secondTroughTime);

        const peakData = data.filter(item => {
            const itemTime = new Date(item.Date).getTime();
            return itemTime > minTime && itemTime < maxTime;
        });

        const peak = peakData.reduce((max, item) => item.High > max.High ? item : max, peakData[0] || data[0]);

        return [
            { time: firstTrough.Date as Time, value: firstTrough.Low },
            { time: peak.Date as Time, value: peak.High },
            { time: secondTrough.Date as Time, value: secondTrough.Low }
        ];
    };

    const getHeadAndShouldersTopPoints = (
        data: OHLCData[],
        startTime: number,
        endTime: number
    ): { time: Time, value: number }[] => {
        if (data.length < 5) return [];

        const rangeData = data.filter(d => {
            const t = new Date(d.Date).getTime();
            return t >= startTime && t <= endTime;
        });

        let bestPattern: { left: OHLCData, head: OHLCData, right: OHLCData, valley1: OHLCData, valley2: OHLCData } | null = null;
        let bestScore = -Infinity;

        for (let i = 2; i < rangeData.length - 2; i++) {
            const left = rangeData[i - 2];
            const valley1 = rangeData[i - 1];
            const head = rangeData[i];
            const valley2 = rangeData[i + 1];
            const right = rangeData[i + 2];

            if (
                left.High < head.High &&
                right.High < head.High &&
                Math.abs(left.High - right.High) / head.High < 0.05 &&
                valley1.Low < left.High &&
                valley2.Low < right.High
            ) {
                const symmetryScore = -Math.abs(new Date(right.Date).getTime() - new Date(head.Date).getTime() - (new Date(head.Date).getTime() - new Date(left.Date).getTime()));
                const shoulderLevel = (left.High + right.High) / 2;
                const score = shoulderLevel - valley1.Low + shoulderLevel - valley2.Low + symmetryScore;

                if (score > bestScore) {
                    bestScore = score;
                    bestPattern = { left, valley1, head, valley2, right };
                }
            }
        }

        if (!bestPattern) return [];

        return [
            { time: bestPattern.left.Date as Time, value: bestPattern.left.High },
            { time: bestPattern.valley1.Date as Time, value: bestPattern.valley1.Low },
            { time: bestPattern.head.Date as Time, value: bestPattern.head.High },
            { time: bestPattern.valley2.Date as Time, value: bestPattern.valley2.Low },
            { time: bestPattern.right.Date as Time, value: bestPattern.right.High }
        ];
    };

    const getHeadAndShouldersBottomPoints = (
        data: OHLCData[],
        startTime: number,
        endTime: number
    ): { time: Time, value: number }[] => {
        if (data.length < 5) return [];

        const rangeData = data.filter(d => {
            const t = new Date(d.Date).getTime();
            return t >= startTime && t <= endTime;
        });

        let bestPattern: { left: OHLCData, head: OHLCData, right: OHLCData, peak1: OHLCData, peak2: OHLCData } | null = null;
        let bestScore = -Infinity;

        for (let i = 2; i < rangeData.length - 2; i++) {
            const left = rangeData[i - 2];
            const peak1 = rangeData[i - 1];
            const head = rangeData[i];
            const peak2 = rangeData[i + 1];
            const right = rangeData[i + 2];

            if (
                left.Low > head.Low &&
                right.Low > head.Low &&
                Math.abs(left.Low - right.Low) / head.Low < 0.05 &&
                peak1.High > left.Low &&
                peak2.High > right.Low
            ) {
                const symmetryScore = -Math.abs(new Date(right.Date).getTime() - new Date(head.Date).getTime() - (new Date(head.Date).getTime() - new Date(left.Date).getTime()));
                const shoulderLevel = (left.Low + right.Low) / 2;
                const score = peak1.High - shoulderLevel + peak2.High - shoulderLevel + symmetryScore;

                if (score > bestScore) {
                    bestScore = score;
                    bestPattern = { left, peak1, head, peak2, right };
                }
            }
        }

        if (!bestPattern) return [];

        return [
            { time: bestPattern.left.Date as Time, value: bestPattern.left.Low },
            { time: bestPattern.peak1.Date as Time, value: bestPattern.peak1.High },
            { time: bestPattern.head.Date as Time, value: bestPattern.head.Low },
            { time: bestPattern.peak2.Date as Time, value: bestPattern.peak2.High },
            { time: bestPattern.right.Date as Time, value: bestPattern.right.Low }
        ];
    };


    const getTrianglePoints = (data: OHLCData[], startTime: number, endTime: number) => {
        if (data.length < 4) return [];

        // Find resistance line (upper trend line)
        const highs = data.map(item => ({ time: item.Date, value: item.High }));
        const resistancePoints = highs.filter((_, index) => index % Math.floor(highs.length / 4) === 0);

        // Find support line (lower trend line)
        const lows = data.map(item => ({ time: item.Date, value: item.Low }));
        const supportPoints = lows.filter((_, index) => index % Math.floor(lows.length / 4) === 0);

        return [...resistancePoints.slice(0, 2), ...supportPoints.slice(0, 2)];
    };

    const getChannelPoints = (data: OHLCData[], startTime: number, endTime: number) => {
        if (data.length < 2) return [];

        const firstPoint = data[0];
        const lastPoint = data[data.length - 1];
        const midPoint = data[Math.floor(data.length / 2)];

        return [
            { time: firstPoint.Date as Time, value: firstPoint.High },
            { time: lastPoint.Date as Time, value: lastPoint.High },
            { time: firstPoint.Date as Time, value: firstPoint.Low },
            { time: lastPoint.Date as Time, value: lastPoint.Low }
        ];
    };

    const getFlagPoints = (data: OHLCData[], startTime: number, endTime: number) => {
        if (data.length < 2) return [];

        // Simple flag pattern - parallel lines
        const slope = (data[data.length - 1].Close - data[0].Close) / data.length;
        return data.map((item, index) => ({
            time: item.Date as Time,
            value: data[0].Close + (slope * index)
        }));
    };

    const getTrendLinePoints = (data: OHLCData[], startTime: number, endTime: number) => {
        if (data.length < 2) return [];

        return [
            { time: data[0].Date as Time, value: data[0].Close },
            { time: data[data.length - 1].Date as Time, value: data[data.length - 1].Close }
        ];
    };

    const drawPatternVisualization = (pattern: Pattern, color: string, data: OHLCData[]) => {
        if (!chartRef.current) return;

        const points = getPatternPoints(pattern, data);
        if (points.length === 0) return;

        const patternType = pattern["Chart Pattern"].toLowerCase();

        if (patternType.includes('double top') || patternType.includes('double bottom')) {
            if (points.length >= 3) {
                // Sort points by time first
                const sortedPoints = [...points].sort((a, b) => {
                    const timeA = typeof a.time === 'string' ? new Date(a.time).getTime() : Number(a.time);
                    const timeB = typeof b.time === 'string' ? new Date(b.time).getTime() : Number(b.time);
                    return Number(timeA) - Number(timeB);
                });

                // Draw first line: first point to valley
                const firstLine = chartRef.current.addLineSeries({
                    color: color,
                    lineWidth: 2,
                    lineStyle: LineStyle.Solid,
                    crosshairMarkerVisible: false,
                    priceLineVisible: false,
                    lastValueVisible: false,
                });

                firstLine.setData([
                    sortedPoints[0],  // First peak/trough
                    sortedPoints[1]   // Valley/peak
                ] as LineData[]);

                patternSeriesRef.current.push(firstLine);

                // Draw second line: valley to second point
                const secondLine = chartRef.current.addLineSeries({
                    color: color,
                    lineWidth: 2,
                    lineStyle: LineStyle.Solid,
                    crosshairMarkerVisible: false,
                    priceLineVisible: false,
                    lastValueVisible: false,
                });

                secondLine.setData([
                    sortedPoints[1],  // Valley/peak
                    sortedPoints[2]   // Second peak/trough
                ] as LineData[]);

                patternSeriesRef.current.push(secondLine);

                // Add horizontal resistance/support line
                if (candlestickSeriesRef.current) {
                    const level = (sortedPoints[0].value + sortedPoints[2].value) / 2;

                    const resistanceLine = candlestickSeriesRef.current.createPriceLine({
                        price: level,
                        color: color,
                        lineWidth: 1,
                        lineStyle: LineStyle.Dashed,
                        axisLabelVisible: true,
                        title: `${pattern["Chart Pattern"]} Level`,
                    });
                    priceLinesRef.current.push(resistanceLine);
                }
            }
        } else if (patternType.includes('triangle')) {
            // Draw triangle pattern with two trend lines
            if (points.length >= 4) {
                // Upper trend line
                const upperLineSeries = chartRef.current.addLineSeries({
                    color: color,
                    lineWidth: 2,
                    lineStyle: LineStyle.Solid,
                    crosshairMarkerVisible: false,
                    priceLineVisible: false,
                    lastValueVisible: false,
                });

                upperLineSeries.setData([points[0], points[1]] as LineData[]);
                patternSeriesRef.current.push(upperLineSeries);

                // Lower trend line
                const lowerLineSeries = chartRef.current.addLineSeries({
                    color: color,
                    lineWidth: 2,
                    lineStyle: LineStyle.Solid,
                    crosshairMarkerVisible: false,
                    priceLineVisible: false,
                    lastValueVisible: false,
                });

                lowerLineSeries.setData([points[2], points[3]] as LineData[]);
                patternSeriesRef.current.push(lowerLineSeries);
            }
        } else if (patternType.includes('channel')) {
            // Draw parallel channel lines
            if (points.length >= 4) {
                // Upper channel line
                const upperLineSeries = chartRef.current.addLineSeries({
                    color: color,
                    lineWidth: 2,
                    lineStyle: LineStyle.Solid,
                    crosshairMarkerVisible: false,
                    priceLineVisible: false,
                    lastValueVisible: false,
                });

                upperLineSeries.setData([points[0], points[1]] as LineData[]);
                patternSeriesRef.current.push(upperLineSeries);

                // Lower channel line
                const lowerLineSeries = chartRef.current.addLineSeries({
                    color: color,
                    lineWidth: 2,
                    lineStyle: LineStyle.Solid,
                    crosshairMarkerVisible: false,
                    priceLineVisible: false,
                    lastValueVisible: false,
                });

                lowerLineSeries.setData([points[2], points[3]] as LineData[]);
                patternSeriesRef.current.push(lowerLineSeries);
            }
        } else {
            // Default: draw a simple trend line
            if (points.length >= 2) {
                const lineSeries = chartRef.current.addLineSeries({
                    color: color,
                    lineWidth: 2,
                    lineStyle: LineStyle.Solid,
                    crosshairMarkerVisible: false,
                    priceLineVisible: false,
                    lastValueVisible: false,
                });

                lineSeries.setData(points as LineData[]);
                patternSeriesRef.current.push(lineSeries);
            }
        }
    };

    useEffect(() => {
        if (chartContainerRef.current) {
            const chart = createChart(chartContainerRef.current, {
                width: chartContainerRef.current.clientWidth,
                height: 500,
                layout: {
                    background: { color: "#1e1e1e" },
                    textColor: "#d1d4dc",
                },
                grid: {
                    vertLines: { color: "#2B2B43" },
                    horzLines: { color: "#2B2B43" },
                },
                timeScale: {
                    timeVisible: true,
                    secondsVisible: false,
                },
                crosshair: {
                    mode: 1,
                },
            });

            const candlestickSeries = chart.addCandlestickSeries({
                upColor: "#26a69a",
                downColor: "#ef5350",
                borderVisible: false,
                wickUpColor: "#26a69a",
                wickDownColor: "#ef5350",
            });

            chartRef.current = chart;
            candlestickSeriesRef.current = candlestickSeries;

            const handleResize = () => {
                if (chartRef.current && chartContainerRef.current) {
                    chartRef.current.applyOptions({
                        width: chartContainerRef.current.clientWidth,
                    });
                }
            };

            window.addEventListener("resize", handleResize);

            // Load initial data
            loadAllData(symbol);

            return () => {
                window.removeEventListener("resize", handleResize);
                clearPatternVisualizations();
                if (chartRef.current) {
                    chartRef.current.remove();
                }
            };
        }
    }, []);

    const detectPatternsInRange = async (startDate: Date, endDate: Date) => {
        setLoading(true);
        setError("");
        setDetectionProgress(0);
        setDetectionStatus('processing');
        // Update visible range with the new selection
        setVisibleRange({ from: startDate, to: endDate });

        // Generate a unique request ID for this pattern detection
        const requestId = Date.now().toString();

        try {
            // Convert pattern thresholds to array format expected by the backend
            const patternThresholdsArray = [
                advancedSettings.pattern_thresholds.double_top,
                advancedSettings.pattern_thresholds.double_bottom,
                advancedSettings.pattern_thresholds.triangle_symmetrical,
                advancedSettings.pattern_thresholds.head_and_shoulders_top,
                advancedSettings.pattern_thresholds.head_and_shoulders_bottom,
                advancedSettings.pattern_thresholds.flag_high_and_tight
            ];

            console.log('Sending settings to backend:', {
                ...advancedSettings,
                probab_threshold_list: patternThresholdsArray
            });

            // Start the pattern detection process
            const startResponse = await axios.get<{
                request_id: string;
                status: string;
            }>(`${API_BASE_URL}/detect-patterns/start/`, {
                params: {
                    symbol,
                    start_date: startDate.toISOString().split('T')[0],
                    end_date: endDate.toISOString().split('T')[0],
                    min_window_size: advancedSettings.min_window_size,
                    max_window_size: advancedSettings.max_window_size,
                    padding_proportion: advancedSettings.padding_proportion,
                    stride: advancedSettings.stride,
                    prob_threshold_of_no_pattern: advancedSettings.prob_threshold_of_no_pattern,
                    min_eps: advancedSettings.min_eps,
                    max_eps: advancedSettings.max_eps,
                    min_samples: advancedSettings.min_samples,
                    iou_threshold: advancedSettings.iou_threshold,
                    probability_difference_threshold: advancedSettings.probability_difference_threshold,
                    final_avg_probability_threshold: advancedSettings.final_avg_probability_threshold,
                    probab_threshold_list: JSON.stringify(patternThresholdsArray)
                },
            });

            // Set up polling for progress updates
            const pollInterval = setInterval(async () => {
                try {
                    const progressResponse = await axios.get<{
                        progress: number;
                        status: string;
                        message?: string;
                        is_complete: boolean;
                        patterns?: Pattern[];
                        error?: string;
                    }>(`${API_BASE_URL}/detect-patterns/progress/${startResponse.data.request_id}`);

                    if (progressResponse.data.error) {
                        throw new Error(progressResponse.data.error);
                    }

                    // Clear existing pattern visualizations
                    clearPatternVisualizations();

                    if (candlestickSeriesRef.current && chartRef.current) {
                        // Get data for the pattern range
                        const patternRangeData = allData.filter(item => {
                            const itemDate = new Date(item.Date);
                            return itemDate >= startDate && itemDate <= endDate;
                        });

                        // Draw pattern visualizations
                        if (progressResponse.data.patterns) {
                            progressResponse.data.patterns.forEach((pattern: Pattern, index: number) => {
                                const color = PATTERN_COLORS[index % PATTERN_COLORS.length];

                                // Get data specific to this pattern's timeframe
                                const patternStartTime = new Date(pattern.Start).getTime();
                                const patternEndTime = new Date(pattern.End).getTime();

                                const patternSpecificData = allData.filter(item => {
                                    const itemTime = new Date(item.Date).getTime();
                                    return itemTime >= patternStartTime && itemTime <= patternEndTime;
                                });

                                drawPatternVisualization(pattern, color, patternSpecificData);
                            });
                        }

                        // Add markers for pattern start/end points
                        if (progressResponse.data.patterns) {
                            const markers: ChartMarker[] = progressResponse.data.patterns.flatMap((pattern: Pattern, index: number) => {
                                const startTime = new Date(pattern.Start).getTime() / 1000 as Time;
                                const endTime = new Date(pattern.End).getTime() / 1000 as Time;
                                const color = PATTERN_COLORS[index % PATTERN_COLORS.length];

                                return [
                                    {
                                        time: startTime,
                                        position: "aboveBar" as const,
                                        color: color,
                                        shape: "arrowDown" as const,
                                        text: `${pattern["Chart Pattern"]} Start`,
                                    },
                                    {
                                        time: endTime,
                                        position: "aboveBar" as const,
                                        color: color,
                                        shape: "arrowUp" as const,
                                        text: `${pattern["Chart Pattern"]} End`,
                                    },
                                ];
                            }).sort((a, b) => Number(a.time) - Number(b.time));

                            candlestickSeriesRef.current.setMarkers(markers);
                        }
                    }

                    // Update progress and status
                    setDetectionProgress(progressResponse.data.progress);
                    setDetectionStatus(progressResponse.data.message || progressResponse.data.status);

                    // If the process is complete, clear the interval and process the results
                    if (progressResponse.data.is_complete) {
                        clearInterval(pollInterval);

                        // Process the patterns if any were found
                        if (progressResponse.data.patterns) {
                            const patternsData = Array.isArray(progressResponse.data.patterns)
                                ? progressResponse.data.patterns
                                : [];
                            setPatterns(patternsData);

                            if (candlestickSeriesRef.current && patternsData.length > 0) {
                                const markers: ChartMarker[] = patternsData.flatMap((pattern: Pattern, index: number) => {
                                    const startTime = new Date(pattern.Start).getTime() / 1000 as Time;
                                    const endTime = new Date(pattern.End).getTime() / 1000 as Time;
                                    const color = PATTERN_COLORS[index % PATTERN_COLORS.length];

                                    return [
                                        {
                                            time: startTime,
                                            position: "aboveBar" as const,
                                            color: color,
                                            shape: "arrowDown" as const,
                                            text: `${pattern["Chart Pattern"]} Start`,
                                        },
                                        {
                                            time: endTime,
                                            position: "aboveBar" as const,
                                            color: color,
                                            shape: "arrowUp" as const,
                                            text: `${pattern["Chart Pattern"]} End`,
                                        },
                                    ];
                                }).sort((a, b) => Number(a.time) - Number(b.time));

                                candlestickSeriesRef.current.setMarkers(markers);
                            } else if (candlestickSeriesRef.current) {
                                // Clear markers if no patterns found
                                candlestickSeriesRef.current.setMarkers([]);
                            }
                        }
                        setLoading(false);
                    }
                } catch (err) {
                    console.error("Error polling progress:", err);
                    clearInterval(pollInterval);
                    const errorMessage = err instanceof Error ? err.message : "Error checking progress";
                    setError(errorMessage);
                    setDetectionStatus('error');
                    setPatterns([]);
                    if (candlestickSeriesRef.current) {
                        candlestickSeriesRef.current.setMarkers([]);
                    }
                    setLoading(false);
                }
            }, 1000); // Poll every second

            // Clean up the interval if the component unmounts
            return () => clearInterval(pollInterval);

        } catch (err) {
            console.error("Error detecting patterns:", err);
            const errorMessage = err instanceof Error ? err.message : "Error detecting patterns";
            setError(errorMessage);
            setDetectionStatus('error');
            // Clear patterns and markers on error
            setPatterns([]);
            if (candlestickSeriesRef.current) {
                candlestickSeriesRef.current.setMarkers([]);
            }
            setLoading(false);
        }
    };

    return (
        <div className="space-y-6">
            <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
                <div className="flex flex-wrap gap-4 items-center">
                    <div className="flex-1 min-w-[200px]">
                        <label htmlFor="symbol" className="block text-sm font-medium text-gray-300 mb-1">
                            Symbol
                        </label>
                        <Combobox value={symbol} onChange={handleSymbolChange}>
                            <div className="relative">
                                <Combobox.Input
                                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    onChange={(event: React.ChangeEvent<HTMLInputElement>) => setQuery(event.target.value)}
                                    displayValue={(symbol: string) => symbol}
                                />
                                <Combobox.Options className="absolute z-10 w-full mt-1 bg-gray-700 rounded-md shadow-lg max-h-60 overflow-auto">
                                    {filteredSymbols.map((symbol) => (
                                        <Combobox.Option
                                            key={symbol}
                                            value={symbol}
                                            className={({ active }: { active: boolean }) =>
                                                `px-3 py-2 cursor-pointer ${active ? 'bg-blue-600 text-white' : 'text-gray-300'
                                                }`
                                            }
                                        >
                                            {symbol}
                                        </Combobox.Option>
                                    ))}
                                </Combobox.Options>
                            </div>
                        </Combobox>
                    </div>
                </div>
                {error && (
                    <p className="mt-4 text-red-500 text-sm">{error}</p>
                )}
                {loading && (
                    <div className="mt-4 flex justify-center">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                    </div>
                )}
            </div>

            <AdvancedSettings onSettingsChange={handleSettingsChange} />

            <TimelineRangeSelector
                visibleStartDate={visibleRange.from}
                visibleEndDate={visibleRange.to}
                onRangeSelect={detectPatternsInRange}
                isSelecting={loading}
                title="Pattern Detection Range"
                buttonText="Detect Patterns"
            />

            {loading && (
                <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
                    <div className="flex flex-col space-y-2">
                        <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-300">
                                {detectionStatus}
                            </span>
                            <span className="text-sm text-gray-300">{detectionProgress}%</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2">
                            <div
                                className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${detectionProgress}%` }}
                            />
                        </div>
                    </div>
                </div>
            )}

            <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
                <div ref={chartContainerRef} className="w-full h-[500px]" />
            </div>

            {patterns.length > 0 && (
                <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
                    <h2 className="text-xl font-semibold mb-4">Detected Patterns with Visualizations</h2>
                    <div className="space-y-2">
                        {patterns.map((pattern: Pattern, index: number) => (
                            <div
                                key={index}
                                className="p-3 bg-gray-700 rounded-md cursor-pointer hover:bg-gray-600 transition"
                                style={{ borderLeft: `4px solid ${PATTERN_COLORS[index % PATTERN_COLORS.length]}` }}
                                onClick={() => zoomToPattern(pattern.Start, pattern.End)}
                            >
                                <div className="flex items-center justify-between">
                                    <span className="font-medium">{pattern["Chart Pattern"]}</span>
                                    <div
                                        className="w-4 h-4 rounded-full"
                                        style={{ backgroundColor: PATTERN_COLORS[index % PATTERN_COLORS.length] }}
                                    />
                                </div>
                                <div className="text-sm text-gray-300 mt-1">
                                    {new Date(pattern.Start).toLocaleDateString()} to{" "}
                                    {new Date(pattern.End).toLocaleDateString()}
                                </div>
                            </div>
                        ))}

                    </div>
                </div>
            )}
        </div>
    );
};

export default ChartContainer;