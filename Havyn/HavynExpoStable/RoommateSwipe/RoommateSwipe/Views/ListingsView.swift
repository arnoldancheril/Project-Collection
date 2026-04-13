//
//  ListingsView.swift
//  RoommateSwipe
//

import SwiftUI

struct ListingsView: View {
    @State private var listings: [ApartmentListing] = []
    @State private var showingAddListingSheet = false
    @State private var selectedListing: ApartmentListing?
    @State private var showingEditSheet = false
    
    var body: some View {
        NavigationView {
            ZStack {
                if listings.isEmpty {
                    VStack(spacing: 20) {
                        Image(systemName: "building.2")
                            .font(.system(size: 60))
                            .foregroundColor(.gray)
                        
                        Text("No Listings Yet")
                            .font(.title2)
                            .foregroundColor(.primary)
                        
                        Text("Add your first property listing to get started")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                        
                        Button(action: {
                            showingAddListingSheet = true
                        }) {
                            HStack {
                                Image(systemName: "plus.circle.fill")
                                Text("Add Listing")
                            }
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                        }
                        .padding(.top, 10)
                    }
                    .padding()
                } else {
                    ScrollView {
                        LazyVStack(spacing: 16) {
                            ForEach(listings) { listing in
                                ListingCard(listing: listing)
                                    .onTapGesture {
                                        selectedListing = listing
                                        showingEditSheet = true
                                    }
                            }
                        }
                        .padding()
                    }
                }
                
                VStack {
                    Spacer()
                    
                    HStack {
                        Spacer()
                        
                        Button(action: {
                            showingAddListingSheet = true
                        }) {
                            Image(systemName: "plus")
                                .font(.title2.weight(.semibold))
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .clipShape(Circle())
                                .shadow(radius: 4)
                        }
                        .padding(.trailing, 20)
                        .padding(.bottom, 20)
                    }
                }
                .opacity(listings.isEmpty ? 0 : 1)
            }
            .navigationTitle("My Listings")
            .onAppear {
                // Load listings - for now using sample data
                listings = ApartmentListing.sampleListings()
            }
            .sheet(isPresented: $showingAddListingSheet) {
                // For now, a placeholder. In real app would use ApartmentListingSignUpView
                Text("Add New Listing")
                    .font(.title)
                    .padding()
            }
            .sheet(item: $selectedListing) { listing in
                // For now, a placeholder. In real app would have a detailed edit view
                Text("Edit \(listing.name)")
                    .font(.title)
                    .padding()
            }
        }
    }
}

struct ListingCard: View {
    let listing: ApartmentListing
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Property image
            ZStack(alignment: .topTrailing) {
                // For now using a placeholder, would use AsyncImage with actual URLs
                Image("exampleProperty1")
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(height: 180)
                    .clipped()
                    .cornerRadius(12)
                
                // Status badge (active/inactive)
                Text(listing.isActive ? "Active" : "Inactive")
                    .font(.caption)
                    .fontWeight(.medium)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(listing.isActive ? Color.green : Color.gray)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                    .padding(8)
            }
            
            // Property details
            VStack(alignment: .leading, spacing: 4) {
                Text(listing.name)
                    .font(.headline)
                
                Text("\(listing.address), \(listing.city)")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                HStack {
                    Text("$\(Int(listing.monthlyRent))/mo")
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundColor(.blue)
                    
                    Spacer()
                    
                    HStack(spacing: 10) {
                        Label("\(listing.bedrooms)", systemImage: "bed.double")
                            .font(.subheadline)
                        
                        Label(String(format: "%.1f", listing.bathrooms), systemImage: "shower")
                            .font(.subheadline)
                    }
                    .foregroundColor(.secondary)
                }
                
                // Interested users count
                HStack {
                    Image(systemName: "person.fill")
                    Text("\(listing.interestedUsers.count) interested")
                    
                    Spacer()
                    
                    // Edit button
                    Button(action: {}) {
                        Text("Manage")
                            .font(.caption)
                            .fontWeight(.medium)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Color(.systemGray6))
                            .foregroundColor(.primary)
                            .cornerRadius(8)
                    }
                }
                .font(.caption)
                .padding(.top, 4)
            }
            .padding(.horizontal, 4)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
    }
}

#Preview {
    ListingsView()
} 