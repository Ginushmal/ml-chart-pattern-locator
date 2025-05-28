'use client';

import React, { useEffect, useRef, useState } from "react";
import { createChart, IChartApi, ISeriesApi, CandlestickData, SeriesMarker, Time } from "lightweight-charts";
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
                    <h2 className="text-xl font-semibold mb-4">Detected Patterns</h2>
                    <div className="space-y-2">
                        {patterns.map((pattern: Pattern, index: number) => (
                            <div
                                key={index}
                                className="p-3 bg-gray-700 rounded-md"
                                style={{ borderLeft: `4px solid ${PATTERN_COLORS[index % PATTERN_COLORS.length]}` }}
                            >
                                <span className="font-medium">{pattern["Chart Pattern"]}</span> -{" "}
                                {new Date(pattern.Start).toLocaleDateString()} to{" "}
                                {new Date(pattern.End).toLocaleDateString()}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default ChartContainer; 