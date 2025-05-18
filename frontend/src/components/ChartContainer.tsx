'use client';

import React, { useEffect, useRef, useState } from "react";
import { createChart, IChartApi, ISeriesApi, CandlestickData, SeriesMarker, Time } from "lightweight-charts";
import axios from "axios";
import { Combobox } from '@headlessui/react';
import TimelineRangeSelector from './TimelineRangeSelector';

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
    const [visibleRange, setVisibleRange] = useState<{ from: Date; to: Date }>({
        from: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000), // One year ago
        to: new Date()
    });

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

        try {
            const response = await axios.get<Pattern[]>(`${API_BASE_URL}/detect-patterns/`, {
                params: {
                    symbol,
                    start_date: startDate.toISOString().split('T')[0],
                    end_date: endDate.toISOString().split('T')[0],
                },
            });

            setPatterns(response.data);

            if (candlestickSeriesRef.current) {
                const markers: ChartMarker[] = response.data.flatMap((pattern: Pattern, index: number) => {
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
        } catch (err) {
            console.error("Error detecting patterns:", err);
            setError(err instanceof Error ? err.message : "Error detecting patterns");
        } finally {
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

            <TimelineRangeSelector
                visibleStartDate={allData.length > 0 ? new Date(allData[0].Date) : new Date()}
                visibleEndDate={allData.length > 0 ? new Date(allData[allData.length - 1].Date) : new Date()}
                onRangeSelect={detectPatternsInRange}
                isSelecting={loading}
                title="Pattern Detection Range"
                buttonText="Detect Patterns"
            />

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