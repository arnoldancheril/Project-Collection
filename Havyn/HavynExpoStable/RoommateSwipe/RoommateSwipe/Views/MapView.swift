//
//  MapView.swift
//  RoommateSwipe
//

import SwiftUI
import MapKit

struct MapView: View {
    @EnvironmentObject var viewModel: RoommateViewModel
    @State private var selectedProfile: Profile?
    @State private var showPropertyDetails = false
    @State private var region = MKCoordinateRegion(
        center: CLLocationCoordinate2D(latitude: 41.8781, longitude: -87.6298), // Chicago coordinates
        span: MKCoordinateSpan(latitudeDelta: 0.1, longitudeDelta: 0.1)
    )
    @State private var selectedFilter: MapFilter?
    @State private var priceRange: ClosedRange<Int> = 500...3000
    @State private var selectedDistance: Int = 5
    @State private var selectedPropertyType: String = "All"
    @State private var selectedNeighborhood: String = "All"
    @State private var showFilterSheet = false
    
    // Demo filters
    private let propertyTypes = ["All", "Apartment", "House", "Condo", "Shared Space"]
    private let neighborhoods = ["All", "Loop", "Wicker Park", "Lincoln Park", "River North", "Hyde Park", "Lakeview"]
    private let distanceOptions = [1, 2, 5, 10, 20]
    
    enum MapFilter: String, CaseIterable, Identifiable {
        case price = "Price"
        case distance = "Distance"
        case propertyType = "Property Type"
        case neighborhood = "Neighborhood"
        
        var id: String { self.rawValue }
        
        var icon: String {
            switch self {
            case .price: return "dollarsign.circle.fill"
            case .distance: return "ruler.circle.fill"
            case .propertyType: return "house.circle.fill"
            case .neighborhood: return "mappin.circle.fill"
            }
        }
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Map Filter Bar
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 10) {
                        ForEach(MapFilter.allCases) { filter in
                            Button(action: {
                                selectedFilter = filter
                                showFilterSheet = true
                            }) {
                                HStack {
                                    Image(systemName: filter.icon)
                                        .foregroundColor(.white)
                                    Text(filter.rawValue)
                                        .foregroundColor(.white)
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(Color.blue.opacity(0.8))
                                .cornerRadius(20)
                            }
                        }
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                }
                .background(
                    LinearGradient(
                        gradient: Gradient(colors: [Color(.systemGray5), Color(.systemGray6)]),
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                
                // Map View
                Map(coordinateRegion: $region, annotationItems: viewModel.profiles.filter { $0.hasRoom }) { profile in
                    MapAnnotation(coordinate: profile.coordinate) {
                        Button {
                            selectedProfile = profile
                            showPropertyDetails = true
                        } label: {
                            VStack {
                                Image(systemName: "house.fill")
                                    .foregroundColor(.blue)
                                    .background(
                                        Circle()
                                            .fill(.white)
                                            .frame(width: 30, height: 30)
                                    )
                                
                                Text(profile.rent)
                                    .font(.caption)
                                    .padding(4)
                                    .background(.white)
                                    .cornerRadius(4)
                                    .shadow(radius: 2)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Chicago Properties")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        // Reset to Chicago center
                        region = MKCoordinateRegion(
                            center: CLLocationCoordinate2D(latitude: 41.8781, longitude: -87.6298),
                            span: MKCoordinateSpan(latitudeDelta: 0.1, longitudeDelta: 0.1)
                        )
                    } label: {
                        Image(systemName: "location")
                    }
                }
            }
            .sheet(isPresented: $showPropertyDetails) {
                NavigationView {
                    if let profile = selectedProfile {
                        MapPropertyView(profile: profile)
                    }
                }
            }
            .sheet(isPresented: $showFilterSheet) {
                MapFilterSheet(
                    filter: selectedFilter,
                    priceRange: $priceRange,
                    selectedDistance: $selectedDistance,
                    selectedPropertyType: $selectedPropertyType,
                    selectedNeighborhood: $selectedNeighborhood,
                    propertyTypes: propertyTypes,
                    neighborhoods: neighborhoods,
                    distanceOptions: distanceOptions
                )
            }
        }
    }
}

// Filter Sheet for Map View
struct MapFilterSheet: View {
    @Environment(\.dismiss) var dismiss
    var filter: MapView.MapFilter?
    @Binding var priceRange: ClosedRange<Int>
    @Binding var selectedDistance: Int
    @Binding var selectedPropertyType: String
    @Binding var selectedNeighborhood: String
    let propertyTypes: [String]
    let neighborhoods: [String]
    let distanceOptions: [Int]
    
    var body: some View {
        NavigationView {
            VStack {
                switch filter {
                case .price:
                    VStack(alignment: .leading, spacing: 20) {
                        Text("Price Range")
                            .font(.headline)
                        
                        Text("$\(Int(priceRange.lowerBound)) - $\(Int(priceRange.upperBound))")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        
                        RangeSlider(
                            value: $priceRange,
                            bounds: 300...5000,
                            step: 100
                        )
                    }
                    .padding()
                    
                case .distance:
                    VStack(alignment: .leading, spacing: 20) {
                        Text("Distance from Center")
                            .font(.headline)
                        
                        Picker("Distance", selection: $selectedDistance) {
                            ForEach(distanceOptions, id: \.self) { distance in
                                Text("\(distance) mile\(distance == 1 ? "" : "s")").tag(distance)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                    }
                    .padding()
                    
                case .propertyType:
                    VStack(alignment: .leading, spacing: 20) {
                        Text("Property Type")
                            .font(.headline)
                        
                        ForEach(propertyTypes, id: \.self) { type in
                            Button(action: {
                                selectedPropertyType = type
                                dismiss()
                            }) {
                                HStack {
                                    Text(type)
                                        .foregroundColor(.primary)
                                    
                                    Spacer()
                                    
                                    if selectedPropertyType == type {
                                        Image(systemName: "checkmark")
                                            .foregroundColor(.blue)
                                    }
                                }
                                .padding()
                                .background(Color(.secondarySystemBackground))
                                .cornerRadius(10)
                            }
                        }
                    }
                    .padding()
                    
                case .neighborhood:
                    VStack(alignment: .leading, spacing: 20) {
                        Text("Neighborhood")
                            .font(.headline)
                        
                        ForEach(neighborhoods, id: \.self) { neighborhood in
                            Button(action: {
                                selectedNeighborhood = neighborhood
                                dismiss()
                            }) {
                                HStack {
                                    Text(neighborhood)
                                        .foregroundColor(.primary)
                                    
                                    Spacer()
                                    
                                    if selectedNeighborhood == neighborhood {
                                        Image(systemName: "checkmark")
                                            .foregroundColor(.blue)
                                    }
                                }
                                .padding()
                                .background(Color(.secondarySystemBackground))
                                .cornerRadius(10)
                            }
                        }
                    }
                    .padding()
                    
                case nil:
                    Text("Select a filter")
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Button("Apply") {
                    dismiss()
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
                .padding()
            }
            .navigationTitle(filter?.rawValue ?? "Filter")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(leading: Button("Cancel") {
                dismiss()
            })
        }
    }
}

// Custom Range Slider for the price filter
struct RangeSlider: View {
    @Binding var value: ClosedRange<Int>
    let bounds: ClosedRange<Int>
    let step: Int
    
    var body: some View {
        VStack {
            HStack {
                Slider(value: Binding(
                    get: { Double(value.lowerBound) },
                    set: { newValue in
                        let rounded = (Int(newValue) / step) * step
                        value = rounded...value.upperBound
                    }
                ), in: Double(bounds.lowerBound)...Double(value.upperBound))
                
                Text("Min: $\(value.lowerBound)")
                    .frame(width: 80, alignment: .trailing)
            }
            
            HStack {
                Slider(value: Binding(
                    get: { Double(value.upperBound) },
                    set: { newValue in
                        let rounded = (Int(newValue) / step) * step
                        value = value.lowerBound...rounded
                    }
                ), in: Double(value.lowerBound)...Double(bounds.upperBound))
                
                Text("Max: $\(value.upperBound)")
                    .frame(width: 80, alignment: .trailing)
            }
        }
    }
}

#Preview {
    MapView()
        .environmentObject(RoommateViewModel())
} 