//
//  AnalyticsView.swift
//  RoommateSwipe
//

import SwiftUI

struct AnalyticsView: View {
    @State private var selectedTimeFrame: TimeFrame = .week
    @State private var selectedListing: ApartmentListing?
    @State private var listings: [ApartmentListing] = []
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    // Time frame selector
                    Picker("Time Frame", selection: $selectedTimeFrame) {
                        ForEach(TimeFrame.allCases, id: \.self) { timeFrame in
                            Text(timeFrame.rawValue).tag(timeFrame)
                        }
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal)
                    
                    // Property selector
                    if !listings.isEmpty {
                        Picker("Select Property", selection: $selectedListing) {
                            Text("All Properties").tag(nil as ApartmentListing?)
                            ForEach(listings) { listing in
                                Text(listing.name).tag(listing as ApartmentListing?)
                            }
                        }
                        .pickerStyle(.menu)
                        .padding(.horizontal)
                    }
                    
                    // Summary stats
                    SummaryStatsView()
                    
                    // Visitor chart
                    ChartCardView(title: "Profile Views", value: 248, change: 12.5, isPositive: true) {
                        // This is a placeholder for a chart
                        FakeBarChartView()
                    }
                    
                    // Interest chart
                    ChartCardView(title: "Interested Users", value: 32, change: -3.8, isPositive: false) {
                        // This is a placeholder for a chart
                        FakeLineChartView()
                    }
                    
                    // Budget insights
                    InsightCardView(
                        title: "Budget Insights",
                        insights: [
                            "Average budget of interested users: $2,350",
                            "75% of interested users have budgets above your asking price",
                            "Budget range of interested users: $1,950 - $2,800"
                        ]
                    )
                    
                    // Demographics insights
                    InsightCardView(
                        title: "Demographics",
                        insights: [
                            "Age range: 24-38 years",
                            "Most common occupations: Software Engineer, Financial Analyst, Teacher",
                            "Average move-in timeline: 45 days"
                        ]
                    )
                }
                .padding(.vertical)
            }
            .navigationTitle("Analytics")
            .onAppear {
                // Load sample data
                listings = ApartmentListing.sampleListings()
            }
        }
    }
}

// Summary statistics view
struct SummaryStatsView: View {
    var body: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
            StatItem(title: "Total Views", value: "248", icon: "eye.fill", color: .blue)
            StatItem(title: "Interested Users", value: "32", icon: "person.fill", color: .green)
            StatItem(title: "Messages", value: "18", icon: "message.fill", color: .purple)
            StatItem(title: "Scheduled Tours", value: "5", icon: "calendar", color: .orange)
        }
        .padding(.horizontal)
    }
}

// Individual stat item
struct StatItem: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.primary)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
    }
}

// Chart card template
struct ChartCardView<ChartContent: View>: View {
    let title: String
    let value: Int
    let change: Double
    let isPositive: Bool
    let chartContent: ChartContent
    
    init(title: String, value: Int, change: Double, isPositive: Bool, @ViewBuilder chartContent: () -> ChartContent) {
        self.title = title
        self.value = value
        self.change = change
        self.isPositive = isPositive
        self.chartContent = chartContent()
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(title)
                    .font(.headline)
                
                Spacer()
                
                HStack(spacing: 4) {
                    Image(systemName: isPositive ? "arrow.up" : "arrow.down")
                    Text("\(String(format: "%.1f", abs(change)))%")
                }
                .font(.caption)
                .foregroundColor(isPositive ? .green : .red)
            }
            
            Text("\(value)")
                .font(.title)
                .fontWeight(.bold)
            
            chartContent
                .frame(height: 120)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
        .padding(.horizontal)
    }
}

// Insights card
struct InsightCardView: View {
    let title: String
    let insights: [String]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.headline)
            
            ForEach(insights, id: \.self) { insight in
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "circle.fill")
                        .font(.system(size: 6))
                        .foregroundColor(.blue)
                        .padding(.top, 6)
                    
                    Text(insight)
                        .font(.subheadline)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
        .padding(.horizontal)
    }
}

// Fake charts for the UI mockup
struct FakeBarChartView: View {
    let bars = [0.6, 0.3, 0.5, 0.8, 0.4, 0.7, 0.9]
    
    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            ForEach(bars.indices, id: \.self) { index in
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.blue.opacity(0.7))
                    .frame(height: 120 * bars[index])
            }
            .frame(maxWidth: .infinity)
        }
    }
}

struct FakeLineChartView: View {
    var body: some View {
        Path { path in
            let width = UIScreen.main.bounds.width - 80
            let height: CGFloat = 120
            
            let points = [
                CGPoint(x: 0, y: height * 0.5),
                CGPoint(x: width * 0.2, y: height * 0.3),
                CGPoint(x: width * 0.4, y: height * 0.7),
                CGPoint(x: width * 0.6, y: height * 0.4),
                CGPoint(x: width * 0.8, y: height * 0.6),
                CGPoint(x: width, y: height * 0.2)
            ]
            
            path.move(to: points[0])
            for i in 1..<points.count {
                path.addLine(to: points[i])
            }
        }
        .stroke(Color.green, lineWidth: 3)
    }
}

#Preview {
    AnalyticsView()
} 