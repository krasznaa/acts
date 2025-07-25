// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Geometry/BoundarySurfaceFace.hpp"
#include "Acts/Geometry/BoundarySurfaceT.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Geometry/Layer.hpp"
#include "Acts/Geometry/Portal.hpp"
#include "Acts/Geometry/TrackingGeometryVisitor.hpp"
#include "Acts/Geometry/TrackingVolumeVisitorConcept.hpp"
#include "Acts/Geometry/Volume.hpp"
#include "Acts/Material/IVolumeMaterial.hpp"
#include "Acts/Navigation/NavigationDelegate.hpp"
#include "Acts/Navigation/NavigationStream.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Surfaces/SurfaceVisitorConcept.hpp"
#include "Acts/Utilities/BinnedArray.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/TransformRange.hpp"
#include "Acts/Visualization/ViewConfig.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/container/container_fwd.hpp>

namespace Acts {

class GlueVolumesDescriptor;
class VolumeBounds;
template <typename object_t>
struct NavigationOptions;
class IMaterialDecorator;
class ISurfaceMaterial;
class IVolumeMaterial;
class Surface;
class TrackingVolume;
struct GeometryIdentifierHook;
class Portal;
class INavigationPolicy;

/// Interface types of the Gen1 geometry model
/// @note This interface is being replaced, and is subject to removal
/// @{

// master typedefs
using TrackingVolumePtr = std::shared_ptr<const TrackingVolume>;
using MutableTrackingVolumePtr = std::shared_ptr<TrackingVolume>;

using TrackingVolumeBoundaryPtr =
    std::shared_ptr<const BoundarySurfaceT<TrackingVolume>>;
using TrackingVolumeBoundaries = std::vector<TrackingVolumeBoundaryPtr>;

// possible contained
using TrackingVolumeArray = BinnedArray<TrackingVolumePtr>;
using TrackingVolumeVector = std::vector<TrackingVolumePtr>;
using MutableTrackingVolumeVector = std::vector<MutableTrackingVolumePtr>;
using LayerArray = BinnedArray<LayerPtr>;
using LayerVector = std::vector<LayerPtr>;

/// Intersection with @c Layer
using LayerIntersection = std::pair<SurfaceIntersection, const Layer*>;
/// Multi-intersection with @c Layer
using LayerMultiIntersection =
    std::pair<SurfaceMultiIntersection, const Layer*>;

/// BoundarySurface of a volume
using BoundarySurface = BoundarySurfaceT<TrackingVolume>;

/// Intersection with a @c BoundarySurface
/// @note This struct is currently split between a gen1 boundary surface
///       and a gen3 portal but only one of them will be set. This will go away
///       once the gen 1 geometry is removed.
struct BoundaryIntersection {
  SurfaceIntersection intersection;
  const BoundarySurface* boundarySurface;
  const Portal* portal;
};

/// Multi-intersection with a @c BoundarySurface
using BoundaryMultiIntersection =
    std::pair<SurfaceMultiIntersection, const BoundarySurface*>;

/// @}

/// @class TrackingVolume
///
/// Full Volume description used in Tracking,
/// it inherits from Volume to get the geometrical structure.
///
///     A TrackingVolume at navigation level can provide the (layer) material
/// information / internal navigation with in
///     5 different ways:
///
///         --- a) Static confinement of Layers
///         --- b) detached sub volumes
///         --- b) unordered (arbitrarily oriented) layers
///         --- d) unordered sub volumes
///         --- e) unordered layers AND unordered subvolumes
///
///    The TrackingVolume can also be a simple container of other
///    TrackingVolumes
///
/// In addition it is capable of holding a subarray of Layers and
/// TrackingVolumes.
///
class TrackingVolume : public Volume {
  friend class TrackingGeometry;

 public:
  TrackingVolume() = delete;
  ~TrackingVolume() override;
  TrackingVolume(const TrackingVolume&) = delete;
  TrackingVolume& operator=(const TrackingVolume&) = delete;
  TrackingVolume(TrackingVolume&&) noexcept;
  TrackingVolume& operator=(TrackingVolume&&) noexcept;

  /// Constructor for a container Volume
  /// - vacuum filled volume either as a for other tracking volumes
  ///
  /// @param transform is the global 3D transform to position the volume in
  /// space
  /// @param volbounds is the description of the volume boundaries
  /// @param volumeName is a string identifier
  TrackingVolume(const Transform3& transform,
                 std::shared_ptr<VolumeBounds> volbounds,
                 const std::string& volumeName = "undefined");

  /// Constructor for a full equipped Tracking Volume
  ///
  /// @param transform is the global 3D transform to position the volume in
  /// space
  /// @param volumeBounds is the description of the volume boundaries
  /// @param volumeMaterial is are materials of the tracking volume
  /// @param staticLayerArray is the confined layer array (optional)
  /// @param containedVolumeArray are the sub volumes if the volume is a
  /// container
  /// @param denseVolumeVector  The contained dense volumes
  /// @param volumeName is a string identifier
  TrackingVolume(
      const Transform3& transform, std::shared_ptr<VolumeBounds> volumeBounds,
      std::shared_ptr<const IVolumeMaterial> volumeMaterial,
      std::unique_ptr<const LayerArray> staticLayerArray = nullptr,
      std::shared_ptr<const TrackingVolumeArray> containedVolumeArray = nullptr,
      MutableTrackingVolumeVector denseVolumeVector = {},
      const std::string& volumeName = "undefined");

  /// Constructor from a regular volume
  /// @param volume is the volume to be converted
  /// @param volumeName is a string identifier
  explicit TrackingVolume(Volume& volume,
                          const std::string& volumeName = "undefined");

  /// Return the associated sub Volume, returns THIS if no subVolume exists
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param position is the global position associated with that search
  /// @param tol Search position tolerance for dense volumes
  ///
  /// @return plain pointer to associated with the position
  const TrackingVolume* lowestTrackingVolume(const GeometryContext& gctx,
                                             const Vector3& position,
                                             const double tol = 0.) const;

  /// @brief Visit all reachable surfaces
  ///
  /// @tparam visitor_t Type of the callable visitor
  ///
  /// @param visitor The callable. Will be called for each reachable surface
  /// that is found, a selection of the surfaces can be done in the visitor
  /// @param restrictToSensitives If true, only sensitive surfaces are visited
  ///
  /// @note If a context is needed for the visit, the vistitor has to provide
  /// this, e.g. as a private member
  template <SurfaceVisitor visitor_t>
  void visitSurfaces(visitor_t&& visitor, bool restrictToSensitives) const {
    auto sensitive = [&visitor](const Surface& surface) {
      if (surface.geometryId().sensitive() == 0) {
        return;
      }
      visitor(&surface);
    };

    if (restrictToSensitives) {
      apply(sensitive);
    } else {
      apply(overloaded{
          sensitive,
          [&visitor](const Portal& portal) { visitor(&portal.surface()); },
          [&visitor](const BoundarySurface& bs) {
            visitor(&bs.surfaceRepresentation());
          }});
    }
  }

  /// @brief Visit all sensitive surfaces
  ///
  /// @tparam visitor_t Type of the callable visitor
  ///
  /// @param visitor The callable. Will be called for each sensitive surface
  /// that is found, a selection of the surfaces can be done in the visitor
  ///
  /// @note If a context is needed for the visit, the vistitor has to provide
  /// this, e.g. as a private member
  template <SurfaceVisitor visitor_t>
  void visitSurfaces(visitor_t&& visitor) const {
    visitSurfaces(std::forward<visitor_t>(visitor), true);
  }

  /// @brief Visit all reachable tracking volumes
  ///
  /// @tparam visitor_t Type of the callable visitor
  ///
  /// @param visitor The callable. Will be called for each reachable volume
  /// that is found, a selection of the volumes can be done in the visitor
  ///
  /// @note If a context is needed for the visit, the vistitor has to provide
  /// this, e.g. as a private member
  template <TrackingVolumeVisitor visitor_t>
  void visitVolumes(visitor_t&& visitor) const {
    apply([&visitor](const TrackingVolume& volume) { visitor(&volume); });
  }

  /// @brief Apply a visitor to the tracking volume
  ///
  /// @param visitor The visitor to apply
  ///
  void apply(TrackingGeometryVisitor& visitor) const;

  /// @brief Apply a mutable visitor to the tracking volume
  ///
  /// @param visitor The visitor to apply
  ///
  void apply(TrackingGeometryMutableVisitor& visitor);

  /// @brief Apply an arbitrary callable as a visitor to the tracking volume
  ///
  /// @param callable The callable to apply
  ///
  /// @note The visitor can be overloaded on any of the arguments that
  ///       the methods in @c TrackingGeometryVisitor receive.
  template <typename Callable>
  void apply(Callable&& callable)
    requires(detail::callableWithAnyMutable<Callable>() &&
             !detail::callableWithAnyConst<Callable>())
  {
    detail::TrackingGeometryLambdaMutableVisitor visitor{
        std::forward<Callable>(callable)};
    apply(visitor);
  }

  /// @brief Apply an arbitrary callable as a visitor to the tracking volume
  ///
  /// @param callable The callable to apply
  ///
  /// @note The visitor can be overloaded on any of the arguments that
  ///       the methods in @c TrackingGeometryMutableVisitor receive.
  template <typename Callable>
  void apply(Callable&& callable) const
    requires(detail::callableWithAnyConst<Callable>())
  {
    detail::TrackingGeometryLambdaVisitor visitor{
        std::forward<Callable>(callable)};
    apply(visitor);
  }

  /// Returns the VolumeName - for debug reason, might be depreciated later
  const std::string& volumeName() const;

  /// Set the volume name to @p volumeName
  /// @param volumeName is the new name of
  void setVolumeName(const std::string& volumeName);

  /// Return the material of the volume
  const IVolumeMaterial* volumeMaterial() const;

  /// Return the material of the volume as shared pointer
  const std::shared_ptr<const IVolumeMaterial>& volumeMaterialPtr() const;

  /// Set the volume material description
  ///
  /// The material is usually derived in a complicated way and loaded from
  /// a framework given source. As various volumes could potentially share the
  /// the same material description, it is provided as a shared object
  ///
  /// @param material Material description of this volume
  void assignVolumeMaterial(std::shared_ptr<const IVolumeMaterial> material);

  /// Return the MotherVolume - if it exists
  const TrackingVolume* motherVolume() const;

  /// Return the MotherVolume - if it exists
  TrackingVolume* motherVolume();

  /// Set the MotherVolume
  ///
  /// @param mvol is the mother volume
  void setMotherVolume(TrackingVolume* mvol);

  using MutableVolumeRange =
      detail::TransformRange<detail::Dereference,
                             std::vector<std::unique_ptr<TrackingVolume>>>;
  using VolumeRange = detail::TransformRange<
      detail::ConstDereference,
      const std::vector<std::unique_ptr<TrackingVolume>>>;

  /// Return all volumes registered under this tracking volume
  /// @return the range of volumes
  VolumeRange volumes() const;

  /// Return mutable view of the registered volumes under this tracking volume
  /// @return the range of volumes
  MutableVolumeRange volumes();

  using MutablePortalRange =
      detail::TransformRange<detail::Dereference,
                             std::vector<std::shared_ptr<Portal>>>;

  using PortalRange =
      detail::TransformRange<detail::ConstDereference,
                             const std::vector<std::shared_ptr<Portal>>>;

  /// Return all portals registered under this tracking volume
  /// @return the range of portals
  PortalRange portals() const;

  /// Return mutable view of the registered portals under this tracking volume
  /// @return the range of portals
  MutablePortalRange portals();

  /// Add a portal to this tracking volume
  /// @param portal The portal to add
  void addPortal(std::shared_ptr<Portal> portal);

  using MutableSurfaceRange =
      detail::TransformRange<detail::Dereference,
                             std::vector<std::shared_ptr<Surface>>>;
  using SurfaceRange =
      detail::TransformRange<detail::ConstDereference,
                             const std::vector<std::shared_ptr<Surface>>>;

  /// Return all surfaces registered under this tracking volume
  /// @return the range of surfaces
  SurfaceRange surfaces() const;

  /// Return mutable view of the registered surfaces under this tracking volume
  /// @return the range of surfaces
  MutableSurfaceRange surfaces();

  /// Add a surface to this tracking volume
  /// @param surface The surface to add
  void addSurface(std::shared_ptr<Surface> surface);

  /// Add a child volume to this tracking volume
  /// @param volume The volume to add
  /// @note The @p volume will have its mother volume assigned to @p this.
  ///       It will throw if @p volume already has a mother volume set
  /// @return Reference to the added volume
  TrackingVolume& addVolume(std::unique_ptr<TrackingVolume> volume);

  /// Interface of @c TrackingVolume in the Gen1 geometry model
  /// @note This interface is being replaced, and is subject to removal
  ///
  /// @{

  /// Return the associated Layer to the global position
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param position is the associated global position
  ///
  /// @return plain pointer to layer object
  const Layer* associatedLayer(const GeometryContext& gctx,
                               const Vector3& position) const;

  /// @brief Resolves the volume into (compatible) Layers
  ///
  /// This is the method for the propagator/extrapolator
  /// @tparam options_t Type of navigation options object for decomposition
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param position Position for the search
  /// @param direction Direction for the search
  /// @param options The templated navigation options
  ///
  /// @return vector of compatible intersections with layers
  boost::container::small_vector<LayerIntersection, 10> compatibleLayers(
      const GeometryContext& gctx, const Vector3& position,
      const Vector3& direction, const NavigationOptions<Layer>& options) const;

  /// @brief Returns all boundary surfaces sorted by the user.
  ///
  /// @tparam options_t Type of navigation options object for decomposition
  /// @tparam sorter_t Type of the boundary surface sorter
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param position The position for searching
  /// @param direction The direction for searching
  /// @param options The templated navigation options
  /// @param logger A @c Logger instance
  ///
  /// @return is the templated boundary intersection
  boost::container::small_vector<BoundaryIntersection, 4> compatibleBoundaries(
      const GeometryContext& gctx, const Vector3& position,
      const Vector3& direction, const NavigationOptions<Surface>& options,
      const Logger& logger = getDummyLogger()) const;

  /// Return the confined static layer array - if it exists
  /// @return the BinnedArray of static layers if exists
  const LayerArray* confinedLayers() const;

  /// Return the confined volumes of this container array - if it exists
  std::shared_ptr<const TrackingVolumeArray> confinedVolumes() const;

  /// Return the confined dense volumes
  const MutableTrackingVolumeVector denseVolumes() const;

  /// Method to return the BoundarySurfaces
  const TrackingVolumeBoundaries& boundarySurfaces() const;

  /// Set the boundary surface material description
  ///
  /// The material is usually derived in a complicated way and loaded from
  /// a framework given source. As various volumes could potentially share the
  /// the same material description, it is provided as a shared object
  ///
  /// @param surfaceMaterial Material description of this volume
  /// @param bsFace Specifies which boundary surface to assign the material to
  void assignBoundaryMaterial(
      std::shared_ptr<const ISurfaceMaterial> surfaceMaterial,
      BoundarySurfaceFace bsFace);

  /// Glue another tracking volume to this one
  ///  - if common face is set the glued volumes are sharing the boundary, down
  /// to the last navigation volume
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param bsfMine is the boundary face indicater where to glue
  /// @param neighbor is the TrackingVolume to be glued
  /// @param bsfNeighbor is the boundary surface of the neighbor
  void glueTrackingVolume(const GeometryContext& gctx,
                          BoundarySurfaceFace bsfMine, TrackingVolume* neighbor,
                          BoundarySurfaceFace bsfNeighbor);

  /// Glue another tracking volume to this one
  ///  - if common face is set the glued volumes are sharing the boundary, down
  /// to the last navigation volume
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param bsfMine is the boundary face indicater where to glue
  /// @param neighbors are the TrackingVolumes to be glued
  /// @param bsfNeighbor are the boundary surface of the neighbors
  void glueTrackingVolumes(
      const GeometryContext& gctx, BoundarySurfaceFace bsfMine,
      const std::shared_ptr<TrackingVolumeArray>& neighbors,
      BoundarySurfaceFace bsfNeighbor);

  /// Provide a new BoundarySurface from the glueing
  ///
  /// @param bsf is the boundary face indicater where to glue
  /// @param bs is the new boundary surface
  /// @param checkmaterial is a flag how to deal with material, if true:
  /// - if the old boundary surface had a material description
  ///   but the new one has not, keep the current one
  /// - in all other cases just assign the new boundary surface
  void updateBoundarySurface(
      BoundarySurfaceFace bsf,
      std::shared_ptr<const BoundarySurfaceT<TrackingVolume>> bs,
      bool checkmaterial = true);

  /// Register the outside glue volumes -
  /// ordering is in the TrackingVolume Frame:
  ///  - negativeFaceXY
  ///  - (faces YZ, ZY, radial faces)
  ///  - positiveFaceXY
  ///
  /// @param gvd register a new GlueVolumeDescriptor
  void registerGlueVolumeDescriptor(std::unique_ptr<GlueVolumesDescriptor> gvd);

  /// Clear boundary surfaces for this tracking volume
  void clearBoundarySurfaces();

  /// Register the outside glue volumes -
  /// ordering is in the TrackingVolume Frame:
  ///  - negativeFaceXY
  ///  - (faces YZ, ZY, radial faces)
  ///  - positiveFaceXY
  GlueVolumesDescriptor& glueVolumesDescriptor();

  /// Produces a 3D visualization of this tracking volume
  /// @param helper The visualization helper describing the output format
  /// @param gctx The geometry context
  /// @param viewConfig The view configuration
  /// @param portalViewConfig View configuration for portals
  /// @param sensitiveViewConfig View configuration for sensitive surfaces
  void visualize(IVisualization3D& helper, const GeometryContext& gctx,
                 const ViewConfig& viewConfig,
                 const ViewConfig& portalViewConfig,
                 const ViewConfig& sensitiveViewConfig) const;

  /// Register a navigation policy with this volume. The argument can not be
  /// nullptr.
  /// @param policy is the navigation policy to be registered
  void setNavigationPolicy(std::unique_ptr<INavigationPolicy> policy);

  /// Populate the navigation stream with navigation candidates from this
  /// volume. Internally, this consults the registered navigation policy, where
  /// the default is a noop.
  /// @param args are the navigation arguments
  /// @param stream is the navigation stream to be updated
  /// @param logger is the logger
  void initializeNavigationCandidates(const NavigationArguments& args,
                                      AppendOnlyNavigationStream& stream,
                                      const Logger& logger) const;

 private:
  void connectDenseBoundarySurfaces(
      MutableTrackingVolumeVector& confinedDenseVolumes);

  /// interlink the layers in this TrackingVolume
  void interlinkLayers();

  /// Create Boundary Surface
  void createBoundarySurfaces();

  /// method to synchronize the layers with potentially updated volume bounds:
  /// - adapts the layer dimensions to the new volumebounds + envelope
  ///
  /// @param envelope is the clearance between volume boundary and layer
  void synchronizeLayers(double envelope = 1.) const;

  // the boundary surfaces
  std::vector<TrackingVolumeBoundaryPtr> m_boundarySurfaces;

  ///(a) static configuration ordered by Binned arrays
  /// static layers
  std::unique_ptr<const LayerArray> m_confinedLayers = nullptr;

  /// Array of Volumes inside the Volume when acting as container
  std::shared_ptr<const TrackingVolumeArray> m_confinedVolumes = nullptr;

  /// confined dense
  MutableTrackingVolumeVector m_confinedDenseVolumes;

  /// Volumes to glue Volumes from the outside
  std::unique_ptr<GlueVolumesDescriptor> m_glueVolumeDescriptor{nullptr};

  /// @}

 private:
  /// The volume based material the TrackingVolume consists of
  std::shared_ptr<const IVolumeMaterial> m_volumeMaterial{nullptr};

  /// Remember the mother volume
  TrackingVolume* m_motherVolume{nullptr};

  /// Volume name for debug reasons & screen output
  std::string m_name;

  std::vector<std::unique_ptr<TrackingVolume>> m_volumes;
  std::vector<std::shared_ptr<Portal>> m_portals;
  std::vector<std::shared_ptr<Surface>> m_surfaces;

  std::unique_ptr<INavigationPolicy> m_navigationPolicy;

  NavigationDelegate m_navigationDelegate{};
};

}  // namespace Acts
