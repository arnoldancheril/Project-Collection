//
//  ListingDetailView.swift
//  RoommateSwipe
//

import SwiftUI
import CoreLocation
import MapKit

struct ListingDetailView: View {
    let listing: ApartmentListing
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var viewModel: RoommateViewModel
    @State private var showingEditSheet = false
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Image carousel
                TabView {
                    ForEach(listing.images, id: \.self) { imageName in
                        if let uiImage = UIImage(named: imageName) {
                            Image(uiImage: uiImage)
                                .resizable()
                                .scaledToFill()
                                .frame(height: 250)
                                .clipped()
                        } else {
                            Image(systemName: "photo")
                                .resizable()
                                .scaledToFit()
                                .frame(height: 250)
                                .foregroundColor(.gray)
                                .background(Color(.systemGray6))
                        }
                    }
                }
                .frame(height: 250)
                .tabViewStyle(PageTabViewStyle())
                .indexViewStyle(PageIndexViewStyle(backgroundDisplayMode: .always))
                
                VStack(alignment: .leading, spacing: 16) {
                    // Title and pricing
                    VStack(alignment: .leading, spacing: 8) {
                        Text(listing.name)
                            .font(.title)
                            .fontWeight(.bold)
                        
                        Text("\(listing.address), \(listing.city), \(listing.state) \(listing.zipCode)")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        
                        HStack {
                            Text("$\(Int(listing.monthlyRent))/month")
                                .font(.title3)
                                .fontWeight(.semibold)
                                .foregroundColor(.blue)
                            
                            Spacer()
                            
                            // Status badge
                            Text(listing.isActive ? "Active" : "Inactive")
                                .font(.caption)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(listing.isActive ? Color.green.opacity(0.2) : Color.red.opacity(0.2))
                                .foregroundColor(listing.isActive ? .green : .red)
                                .cornerRadius(8)
                        }
                    }
                    
                    Divider()
                    
                    // Property details
                    HStack(spacing: 20) {
                        PropertyDetailItem(icon: "bed.double.fill", value: "\(listing.bedrooms)")
                        PropertyDetailItem(icon: "shower.fill", value: String(format: "%.1f", listing.bathrooms))
                        PropertyDetailItem(icon: "square.fill", value: "\(listing.squareFootage) sq ft")
                        PropertyDetailItem(icon: "calendar", value: listing.availableDate.formattedMonthYear)
                    }
                    
                    Divider()
                    
                    // Description
                    VStack(alignment: .leading, spacing: 8) {
                        Text("About this property")
                            .font(.headline)
                        
                        Text(listing.description)
                            .font(.body)
                            .foregroundColor(.secondary)
                    }
                    
                    Divider()
                    
                    // Amenities
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Amenities")
                            .font(.headline)
                        
                        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                            ForEach(listing.amenities, id: \.self) { amenity in
                                HStack(spacing: 8) {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.green)
                                    Text(amenity)
                                        .font(.subheadline)
                                        .foregroundColor(.primary)
                                }
                            }
                        }
                    }
                    
                    Divider()
                    
                    // Pet policy
                    HStack {
                        Image(systemName: listing.petPolicy.icon)
                            .foregroundColor(.blue)
                        Text("Pet Policy: \(listing.petPolicy.rawValue)")
                            .font(.subheadline)
                    }
                    
                    Divider()
                    
                    // Lease length
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Lease Options")
                            .font(.headline)
                        
                        HStack {
                            ForEach(listing.leaseLength, id: \.self) { lease in
                                Text(lease.rawValue)
                                    .font(.caption)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(Color.blue.opacity(0.1))
                                    .foregroundColor(.blue)
                                    .cornerRadius(8)
                            }
                        }
                    }
                    
                    Divider()
                    
                    // Location
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Location")
                            .font(.headline)
                        
                        ListingMapView(coordinate: listing.coordinates)
                            .frame(height: 200)
                            .cornerRadius(12)
                    }
                    
                    // Interested users summary
                    if !listing.interestedUsers.isEmpty {
                        Divider()
                        
                        HStack {
                            Text("\(listing.interestedUsers.count) Interested Users")
                                .font(.headline)
                            
                            Spacer()
                            
                            NavigationLink(destination: InterestedUsersView()) {
                                Text("View All")
                                    .font(.subheadline)
                                    .foregroundColor(.blue)
                            }
                        }
                    }
                }
                .padding()
            }
        }
        .navigationTitle("Listing Details")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarItems(trailing: Button(action: {
            showingEditSheet = true
        }) {
            Text("Edit")
        })
        .sheet(isPresented: $showingEditSheet) {
            NavigationView {
                ListingEditView(listing: listing) { updatedListing in
                    viewModel.updateListing(updatedListing)
                    dismiss()
                }
                .navigationTitle("Edit Listing")
                .navigationBarItems(leading: Button("Cancel") {
                    showingEditSheet = false
                })
            }
        }
    }
}

struct PropertyDetailItem: View {
    let icon: String
    let value: String
    
    var body: some View {
        VStack {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(.blue)
            
            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
        }
        .frame(maxWidth: .infinity)
    }
}

extension Date {
    var formattedMonthYear: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM yyyy"
        return formatter.string(from: self)
    }
}

#Preview {
    NavigationView {
        ListingDetailView(listing: ApartmentListing.sampleListings()[0])
            .environmentObject(RoommateViewModel())
    }
} 