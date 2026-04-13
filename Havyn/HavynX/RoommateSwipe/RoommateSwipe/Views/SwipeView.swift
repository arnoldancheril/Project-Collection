//
//  SwipeView.swift
//  RoommateSwipe
//

import SwiftUI

struct SwipeView: View {
    @EnvironmentObject var viewModel: RoommateViewModel
    @State private var showFilters = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Modern Filter Bar
                FilterBarView(viewModel: viewModel)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(
                        LinearGradient(
                            gradient: Gradient(colors: [
                                Color(.systemGray6),
                                Color(.systemGray6).opacity(0.5),
                                Color(.systemBackground)
                            ]),
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                
                if viewModel.filteredProfiles.isEmpty {
                    Spacer()
                    Text("No more profiles in your area.")
                        .font(.title2)
                        .foregroundColor(.gray)
                        .padding()
                    Spacer()
                } else {
                    GeometryReader { geo in
                        ForEach(viewModel.filteredProfiles.indices, id: \.self) { index in
                            let profile = viewModel.filteredProfiles[index]
                            
                            SwipeCardView(
                                profile: profile,
                                onSwipeLeft: {
                                    viewModel.dislike(profile: profile)
                                    viewModel.removeProfile(profile: profile)
                                },
                                onSwipeRight: {
                                    viewModel.like(profile: profile)
                                    viewModel.removeProfile(profile: profile)
                                }
                            )
                            .stacked(at: index, in: viewModel.filteredProfiles.count)
                            .frame(width: geo.size.width, height: geo.size.height - 30)
                        }
                    }
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .background(Color(.systemBackground))
        }
    }
}

// Modern Filter Bar Component
struct FilterBarView: View {
    @ObservedObject var viewModel: RoommateViewModel
    @State private var selectedFilters: Set<FilterType> = []
    @State private var showFilterSheet = false
    @State private var cityFilter = "Any"
    @State private var budgetRange = 0...5000
    @State private var genderFilter = "All"
    @State private var petFilter = "No Preference"
    @State private var housingFilter = "No Preference"
    @State private var lifestyleFilter = "No Preference"
    @State private var cleanlinessFilter = "No Preference"
    
    enum FilterType: String, CaseIterable {
        case city = "Location"
        case budget = "Budget"
        case gender = "Gender"
        case pets = "Pets"
        case housing = "Housing"
        case lifestyle = "Lifestyle"
        case cleanliness = "Cleanliness"
        
        var icon: String {
            switch self {
            case .city: return "location.circle.fill"
            case .budget: return "dollarsign.circle.fill"
            case .gender: return "person.2.circle.fill"
            case .pets: return "pawprint.circle.fill"
            case .housing: return "house.circle.fill"
            case .lifestyle: return "moon.circle.fill"
            case .cleanliness: return "sparkles.circle.fill"
            }
        }
        
        // Helper to get the current value for this filter type
        func getValue(_ filterView: FilterBarView) -> String {
            switch self {
            case .city: return filterView.cityFilter
            case .budget: return "$\(filterView.budgetRange.lowerBound)-$\(filterView.budgetRange.upperBound)"
            case .gender: return filterView.genderFilter
            case .pets: return filterView.petFilter
            case .housing: return filterView.housingFilter
            case .lifestyle: return filterView.lifestyleFilter
            case .cleanliness: return filterView.cleanlinessFilter
            }
        }
        
        // Helper to check if filter has non-default value
        func isActive(_ filterView: FilterBarView) -> Bool {
            switch self {
            case .city: return filterView.cityFilter != "Any"
            case .budget: return filterView.budgetRange != 0...5000
            case .gender: return filterView.genderFilter != "All"
            case .pets: return filterView.petFilter != "No Preference"
            case .housing: return filterView.housingFilter != "No Preference"
            case .lifestyle: return filterView.lifestyleFilter != "No Preference"
            case .cleanliness: return filterView.cleanlinessFilter != "No Preference"
            }
        }
        
        // Helper to reset this filter type to default
        func reset(_ filterView: FilterBarView) {
            switch self {
            case .city: filterView.cityFilter = "Any"
            case .budget: filterView.budgetRange = 0...5000
            case .gender: filterView.genderFilter = "All"
            case .pets: filterView.petFilter = "No Preference"
            case .housing: filterView.housingFilter = "No Preference"
            case .lifestyle: filterView.lifestyleFilter = "No Preference"
            case .cleanliness: filterView.cleanlinessFilter = "No Preference"
            }
        }
    }
    
    var body: some View {
        VStack(spacing: 12) {
            // Active Filters Display
            ScrollView(.horizontal, showsIndicators: false) {
        HStack(spacing: 8) {
                    ForEach(FilterType.allCases, id: \.self) { filter in
                        FilterChip(
                            type: filter,
                            isSelected: selectedFilters.contains(filter),
                            isActive: filter.isActive(self),
                            value: filter.getValue(self)
                        ) {
                            withAnimation {
                                if selectedFilters.contains(filter) {
                                    selectedFilters.remove(filter)
                                    filter.reset(self) // Reset when deselecting
                                } else {
                                    selectedFilters.insert(filter)
                                }
                                showFilterSheet = true
                            }
                        }
                    }
                }
                .padding(.horizontal, 4)
            }
            
            // Apply and Clear Buttons
            if !selectedFilters.isEmpty {
                HStack(spacing: 12) {
                    Button(action: clearAllFilters) {
                        Text("Clear All")
                            .font(.subheadline)
                            .foregroundColor(.blue)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Color.blue.opacity(0.1))
                            .cornerRadius(20)
                    }
                    
                    Button(action: applyFilters) {
                        Text("Apply")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .padding(.horizontal, 20)
                            .padding(.vertical, 8)
                            .background(Color.blue)
                            .cornerRadius(20)
                            .shadow(color: Color.blue.opacity(0.3), radius: 4, x: 0, y: 2)
                    }
                }
                .transition(.scale.combined(with: .opacity))
            }
        }
        .sheet(isPresented: $showFilterSheet) {
            FilterDetailSheet(
                selectedFilters: $selectedFilters,
                cityFilter: $cityFilter,
                budgetRange: $budgetRange,
                genderFilter: $genderFilter,
                petFilter: $petFilter,
                housingFilter: $housingFilter,
                lifestyleFilter: $lifestyleFilter,
                cleanlinessFilter: $cleanlinessFilter,
                onApply: applyFilters
            )
        }
    }
    
    private func clearAllFilters() {
        for filter in FilterType.allCases {
            filter.reset(self)
        }
        selectedFilters.removeAll()
        applyFilters()
    }
    
    private func applyFilters() {
        // Apply the filters to the viewModel
        viewModel.filterCity = cityFilter == "Any" ? "" : cityFilter
        viewModel.filterGender = genderFilter == "All" ? "No Preference" : genderFilter
        // Add other filter applications here
        
        // Optionally close the filter sheet
        showFilterSheet = false
    }
}

// Filter Chip Component
struct FilterChip: View {
    let type: FilterBarView.FilterType
    let isSelected: Bool
    let isActive: Bool
    let value: String
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: type.icon)
                    .font(.system(size: 14))
                VStack(alignment: .leading, spacing: 2) {
                    Text(type.rawValue)
                        .font(.system(size: 13, weight: .medium))
                    if isActive {
                        Text(value)
                            .font(.system(size: 11))
                            .lineLimit(1)
                    }
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(backgroundColor)
            .foregroundColor(isSelected || isActive ? .white : .blue)
            .cornerRadius(16)
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.blue.opacity(0.2), lineWidth: (!isSelected && !isActive) ? 1 : 0)
            )
        }
    }
    
    private var backgroundColor: Color {
        if isSelected {
            return .blue
        } else if isActive {
            return .blue.opacity(0.8)
        } else {
            return .blue.opacity(0.1)
        }
    }
}

// Filter Detail Sheet
struct FilterDetailSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Binding var selectedFilters: Set<FilterBarView.FilterType>
    @Binding var cityFilter: String
    @Binding var budgetRange: ClosedRange<Int>
    @Binding var genderFilter: String
    @Binding var petFilter: String
    @Binding var housingFilter: String
    @Binding var lifestyleFilter: String
    @Binding var cleanlinessFilter: String
    let onApply: () -> Void
    
    // Predefined cities
    let cities = ["Any", "Chicago", "New York", "Los Angeles", "San Francisco", "Boston", "Seattle"]
    
    var body: some View {
        NavigationView {
            Form {
                if selectedFilters.contains(.city) {
                    Section("Location") {
                        Picker("City", selection: $cityFilter) {
                            ForEach(cities, id: \.self) { city in
                                Text(city).tag(city)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                }
                
                if selectedFilters.contains(.budget) {
                    Section("Budget Range") {
                        HStack {
                            Text("$\(budgetRange.lowerBound)")
                            Slider(
                                value: .init(
                                    get: { Double(budgetRange.lowerBound) },
                                    set: { budgetRange = Int($0)...budgetRange.upperBound }
                                ),
                                in: 0...10000,
                                step: 100
                            )
                        }
                        HStack {
                            Text("$\(budgetRange.upperBound)")
                            Slider(
                                value: .init(
                                    get: { Double(budgetRange.upperBound) },
                                    set: { budgetRange = budgetRange.lowerBound...Int($0) }
                                ),
                                in: 0...10000,
                                step: 100
                            )
                        }
                    }
                }
                
                if selectedFilters.contains(.gender) {
                    Section("Gender Preference") {
                        Picker("Gender", selection: $genderFilter) {
                            Text("All").tag("All")
                            Text("Male").tag("Male")
                            Text("Female").tag("Female")
                            Text("Non-binary").tag("Non-binary")
                        }
                        .pickerStyle(.segmented)
                    }
                }
                
                if selectedFilters.contains(.pets) {
                    Section("Pet Preference") {
                        Picker("Pets", selection: $petFilter) {
                            Text("Any").tag("No Preference")
                            Text("Allowed").tag("Allowed")
                            Text("Not Allowed").tag("Not Allowed")
                        }
                        .pickerStyle(.segmented)
                    }
                }
                
                if selectedFilters.contains(.housing) {
                    Section("Housing Type") {
                        Picker("Housing", selection: $housingFilter) {
                            Text("Any").tag("No Preference")
                            Text("Apartment").tag("Apartment")
                            Text("House").tag("House")
                            Text("Shared").tag("Shared")
                        }
                        .pickerStyle(.segmented)
                    }
                }
                
                if selectedFilters.contains(.lifestyle) {
                    Section("Lifestyle") {
                        Picker("Lifestyle", selection: $lifestyleFilter) {
                            Text("Any").tag("No Preference")
                            Text("Morning").tag("Morning")
                            Text("Night").tag("Night")
                            Text("Flexible").tag("Flexible")
                        }
                        .pickerStyle(.segmented)
                    }
                }
                
                if selectedFilters.contains(.cleanliness) {
                    Section("Cleanliness") {
                        Picker("Cleanliness", selection: $cleanlinessFilter) {
                            Text("Any").tag("No Preference")
                            Text("Neat").tag("Neat")
                            Text("Moderate").tag("Moderate")
                            Text("Relaxed").tag("Relaxed")
                        }
                        .pickerStyle(.segmented)
                    }
                }
            }
            .navigationTitle("Filters")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Apply") {
                        onApply()
                        dismiss()
                    }
                    .fontWeight(.semibold)
                }
            }
        }
    }
}

extension View {
    func stacked(at index: Int, in total: Int) -> some View {
        let offset = Double(total - index) * 4
        return self.offset(x: 0, y: offset)
    }
}

#Preview {
    SwipeView()
        .environmentObject(RoommateViewModel())
}

