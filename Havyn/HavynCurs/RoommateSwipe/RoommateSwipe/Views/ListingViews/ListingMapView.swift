//
//  ListingMapView.swift
//  RoommateSwipe
//

import SwiftUI
import MapKit

struct ListingMapView: View {
    let coordinate: CLLocationCoordinate2D
    @State private var region: MKCoordinateRegion
    
    init(coordinate: CLLocationCoordinate2D) {
        self.coordinate = coordinate
        self._region = State(initialValue: MKCoordinateRegion(
            center: coordinate,
            span: MKCoordinateSpan(latitudeDelta: 0.01, longitudeDelta: 0.01)
        ))
    }
    
    var body: some View {
        Map(coordinateRegion: $region, annotationItems: [MapPin(coordinate: coordinate)]) { pin in
            MapMarker(coordinate: pin.coordinate, tint: .blue)
        }
    }
}

struct MapPin: Identifiable {
    let id = UUID()
    let coordinate: CLLocationCoordinate2D
}

#Preview {
    ListingMapView(coordinate: CLLocationCoordinate2D(latitude: 40.7128, longitude: -74.0060))
} 